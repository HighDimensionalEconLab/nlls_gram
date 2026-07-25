# Pytree metric/preconditioner state and caller-owned adaptation

Design proposal for review. Branch `refactor/unify-solver-contracts`, version
stays 2.7.0. Breaking changes are free: all callers are our own repos
(`kernels`, `spooky`), ported in the same effort. No shims, no deprecations.
Revision 2, incorporating the first external review.

## Problem

The current contract conflates three things:

1. **Construction** — how metric/preconditioner numeric state is built
   (`prepare(theta, ctx)` on each class, plus the `block_eigen_state` free
   function, plus `blocks_fn` closure fields).
2. **Refresh policy** — when it is rebuilt (`rebuild(ctx)` predicates,
   `metric_valid`/`precond_valid` reject-reuse flags, the subclass-override
   idiom for "rebuild on ridge advance").
3. **Hashing** — instances hold arrays and closures, so they identity-hash
   into the solver's static key, and a rebuilt equal-config instance keys a
   fresh compilation of the whole solve loop.

Symptoms: the kernels driver threads preconditioner state through `args` with
~25 lines of callback plumbing; `rebuild` receives a context without the
carried state it would need to decide; four `LMState` fields and three
`lm_core` methods exist only to shepherd hook state around.

## Design

### Instances are pytrees

Every concrete `Metric` and `Preconditioner` class is registered as a JAX
pytree: array fields are traced leaves, the type plus its static fields are
structure. Instances ride **inside `LMState`** (`lm_state.metric`,
`lm_state.preconditioner`) as nested carried state, so their arrays flow
through the jitted loop, `vmap`, and the implicit-AD rule like any other
state.

**Rebuild = call the constructor again.** There is no `prepare`, no
`rebuild`, no `remake`. A callback that wants a fresh preconditioner
constructs a new instance of the same type inside the traced callback — pure
traced ops, same treedef, no recompilation. A different type or field
structure is a loud trace-time error. Constructors must canonicalize their
leaf dtypes (`jnp.asarray(..., dtype)`), so a rebuilt instance's leaf avals
match the carried ones inside the user's `lax.cond`.

`dataclasses.replace` semantics: fine for **metrics** (their `__post_init__`
is validation plus shape derivation, cheap under trace); **preconditioners
are rebuilt by constructor, never `replace`d** — `replace` re-runs
`__init__`, which re-pays the eigendecomposition/sketch, and the
construction-time-only inputs below are not stored to re-supply. Documented
on the base classes.

### Registration helper (public)

`jax.tree_util.register_dataclass` unflattens by calling the constructor,
which re-runs `__post_init__` (re-`eigh`, tracer-hostile validation) on
every carry reconstruction — verified against the installed JAX. So the
package ships its own ~20-line helper in `utilities.py`, exported as public
API:

```python
def register_pytree_dataclass(cls, *, data_fields, meta_fields=()):
    """Register a frozen dataclass as a pytree whose unflatten BYPASSES
    __init__/__post_init__ (object.__new__ + object.__setattr__).
    data_fields are traced leaves; meta_fields are static structure and
    must be hashable. Returns cls."""
```

Consequences:

- Constructors freely compute derived leaves once (`eigh` in
  `BlockEigenPreconditioner`, the sketch in `NystromPreconditioner`) and
  validate eagerly; unflatten restores fields verbatim.
- Aux data is the tuple of static field values passed through the existing
  `_typed_key` type-tagging, so treedefs hash and compare **by value with
  jit's strict-type semantics**: two equal-config instances with fresh
  arrays have equal treedefs, while `1`, `1.0`, and `True` in a static
  field stay distinct (raw tuples would collapse them).
- The helper validates that every dataclass field appears exactly once in
  `data_fields` or `meta_fields`, so a forgotten field is a registration
  error rather than a silently dropped leaf.
- `tree_map` over an instance rebuilds it without re-running the
  constructor, so a transform that changes leaf shapes or breaks a derived
  invariant (`permutation` without `inverse_permutation`) produces an
  inconsistent instance. Documented: shape-changing or invariant-coupled
  mutation goes through the constructor; `tree_map` is for
  transform-machinery (vmap batching, tangent zeroing), which preserves
  both.
- Each concrete class registers itself (subclassing a registered base does
  not register the subclass); the custom-metric and custom-preconditioner
  doc recipes lead with the registration call. The solver raises a clear
  error at construction when handed an unregistered instance (one that
  flattens as a leaf), pointing at the helper.
- `_EuclideanMetric` — the `metric=None` default — is registered too (a
  frozen dataclass with no leaves), so the default path passes the same
  check.

Static/leaf split for the shipped classes (construction-time-only inputs are
`dataclasses.InitVar`s — consumed by `__post_init__`, never stored):

| class | leaves (traced) | static (structure) | InitVar (consumed) |
|---|---|---|---|
| `IdentityMetric` | `free_scale` | `size` | |
| `CholeskyMetric` | `L`, `free_scale` | `size` | |
| `DiagonalMetric` | `weights`, `free_scale` | `size` | |
| `RepeatedFactorMetric` | `F`, `free_scale` | `repeats`, `size` | |
| `IdentityPreconditioner` | — | — | |
| `BlockEigenPreconditioner` | per-family eigenvectors/eigenvalues/ridge_weight, `permutation`, `inverse_permutation` | family count (structure; array shapes live in the leaf avals) | `families` |
| `ShermanMorrisonPreconditioner` | `u`, `weight`, `solve_u`, `denominator` | `solve` | |
| `WoodburyPreconditioner` | `U`, `weights`, `solve_U`, `capacitance_factor` (array only — `cho_factor`'s `lower` bool is passed as a literal in `apply`, never stored as a leaf) | `solve` | |
| `PaddedPreconditioner` | `base` (subtree) | `n_real` | |
| `NystromPreconditioner` | `basis`, `eigenvalues` | `n`, `rank` | `matvec`, `key`, `dtype` |

`ShermanMorrison`/`Woodbury` call `solve` on **every** `apply` (it is the
action of `A^{-1}`), so it cannot be consumed at construction; it is a
**static field**: a fixed hashable callable whose identity enters the
treedef, and whose closed-over arrays are compile-time constants. These two
classes are setup-scope objects for fixed dual operators, not
callback-refresh targets; rebuilding with the *same* `solve` callable keeps
the treedef. `NystromPreconditioner.matvec` genuinely is construction-only
(its `apply` uses only the stored sketch) and stays an `InitVar`.
`experimental.StateSpaceMetric` gets the same deliberate split when it is
registered: `transition` is construction-only, `parallel` static (its ops
branch on it in Python).

`free_scale` is a **leaf**: changing it must not recompile. (`_free_scale`
loses its `scale == 1.0` Python short-circuit — a tracer breaks it — and
always divides.) Constructors canonicalize it to a strongly-typed scalar of
the factor's float dtype (default float when the metric holds no arrays),
so an initial Python `1.0` and a callback-rebuilt value have identical
avals. Validation of possibly-traced fields follows the existing
convention: concrete values are validated eagerly, tracers/arrays skip the
sign checks.

`BlockEigenPreconditioner` drops `blocks_fn` entirely:

```python
BlockEigenPreconditioner(families, permutation)
# families: sequence of (blocks, ridge_weight) pairs, blocks shaped
# (groups, size, size) — the constructor symmetrizes, eigendecomposes,
# and stores the results as leaves (absorbing today's block_eigen_state,
# which is deleted).
```

### Contracts shrink

```python
class Metric:
    size: int          # static
    free_scale         # leaf
    def factor_apply(self, v, ctx): ...
    def factor_solve(self, v, ctx): ...
    def factor_solve_transpose(self, v, ctx): ...
    def norm(self, v, ctx): ...    # defaulted via factor_apply

class Preconditioner:
    requires_positive_damping = False
    def apply(self, v, damping, ctx): ...
```

Ops read `self` — the carried instance. Deleted: `Metric.prepare`,
`Metric.rebuild`, `Preconditioner.prepare`, `Preconditioner.rebuild`,
`block_eigen_state`, `SolverContext.metric_state`,
`SolverContext.preconditioner_state`, `LMState.metric_state`,
`LMState.metric_valid`, `LMState.precond`, `LMState.precond_valid`,
`lm_core`'s `_init_hook_state`/`_hook_state`/`_frozen_ctx`, and the
duplicate `_block_sizes` definition in `lm_core.py`.

`SolverContext` keeps `x`, `lm_state`, `args`, `p` — an exotic metric can
still key off the live iterate, and reaches any carried state through
`ctx.lm_state`.

### Every traced read goes through the carried instance

This is the load-bearing rule, and it covers **`linear_solvers.py`** too:
`CG.prepare` and `GramCG.prepare` currently build `apply_M` from
`self.preconditioner` — a jit-static object whose arrays are compile-time
constants. All such sites move to the carried instances
(`sub.ctx.lm_state.preconditioner`; the metric via the ctx the whiten
closures already close over). To make any stray old-style read fail loudly
instead of silently baking one instance's arrays into a shared compile, the
solver's constructor-held attributes are renamed: `self.initial_metric`,
`self.initial_preconditioner`. They exist to seed `init`/`_cold_state`, to
key compilation, and for static reads (`metric.size` — sound either way,
since the treedef contract pins statics). The config fields
(`CG.preconditioner`) keep their names as the initial-instance source but
are never read inside `prepare`.

The enforcement is a correctness test, not just a convention: two solvers
built around same-treedef, **different-valued** metrics (and
preconditioners) must produce different answers on every linear-solver path
— under the new value-based static key they share one compiled loop, so any
leftover static read reproduces the first solver's geometry and fails this
test. (The old identity-hashed keys made that bug impossible; the new tests
are what make the sharing safe.)

### LMState

```python
@dataclass(frozen=True)
class LMState:
    damping: jax.Array
    ridge: jax.Array | None = None
    resid / Jt / jacobian_valid / aux        # Jacobian cache, unchanged
    hyper: LMHyperparams | None = None
    solver_cache: Any = None
    metric: Metric | None = None             # NEW: the carried instance
    preconditioner: Preconditioner | None = None   # NEW
```

Ownership contract, stated in the docs: **callback-owned** state is
`damping`, `ridge`, `metric`, `preconditioner`, and the `hyper` group
(replaceable as a unit); everything else (`resid`/`Jt`/`jacobian_valid`/
`aux`/`solver_cache`) is **solver-owned** — preserve it with
`dataclasses.replace(ctx.lm_state, ...)`. No deeper nesting: grouping the
live `damping` with its schedule would mix a solver-written scalar with
callback-owned knobs and force nested replaces in the hot update path.

`init` and `_cold_state` seed the constructor instances into the state, and
so does the metric solver's minimal-state fast path in `_solve_lm_state`
(today it returns a bare `LMState(damping)`; that would break the carry
structure) — so every state entering the solve loop carries instances.
Inside the loop every traced read goes through the carried instances.
`update` handed a hand-built state with `metric=None` **reads** through
`initial_metric`/`initial_preconditioner` but **passes the `None`
through** to the returned state — injecting an instance would change the
carry structure of a user's own `lax.while_loop` around `update`. With
`prepare` gone, a manual loop that wants an iterate-tracking metric
rebuilds the instance between its own calls. `_cast_state` casts named
scalars only (`damping`, `ridge`, `hyper`, cache ridge) and never touches
instance leaves.

### All adaptive policy lives in the single per-step callback

Ridge anneal, preconditioner refresh, metric swap, data re-draw, damping
reset, custom stop — one callback, running where it does today: after the
accept/reject update, before the termination test. (Validated against
spooky's `epoch_callback`, which keeps working unchanged modulo the rename
below.)

The solver keeps only the reactive, non-policy automation, all as extensions
of the existing `_apply_action` + `problem_changed` machinery:

1. **Metric mutation is detected automatically by value**, the same rule as
   `x`/`args` today: `_apply_action` compares the action's metric leaves
   against the carried ones with `_tree_changed`, short-circuiting at trace
   time on leaf **identity** — a callback that used
   `dataclasses.replace(ctx.lm_state, ridge=...)` hands back the same
   tracer objects for the metric subtree, so the untouched-metric case
   emits no comparison ops at all (large factors pay the `array_equal`
   reduction only on steps that actually rebuild). What a change *means*
   differs by solver, because the metric's role differs:
   - **`RidgeLevenbergMarquardt`**: the metric defines the objective, so a
     change sets `problem_changed` — suppressing that iteration's
     convergence test (the step's diagnostics were computed under the old
     geometry) — and invalidates the solver caches (`G`/`R` embed the
     whitening).
   - **`LevenbergMarquardt`**: the metric is damping geometry only; the
     objective `||r||^2` did not move, so convergence is **not**
     suppressed. The solver caches are still invalidated (the assembled
     whitened normal matrix embeds the factor).
   - The Jacobian cache is **never** cleared by a metric change: `J = dr/dx`
     does not depend on the metric, and whitening is applied fresh from the
     cached `Jt` each step.

   Consequence worth stating for the ridge solver: an iterate-dependent
   metric rebuilt on *every* accepted step suppresses convergence on every
   accepted step, so `xtol` (accepted-only) can never fire and termination
   comes from `gtol`/`atol` on rejected steps or from a conditional rebuild
   (`lax.cond` on progress). Rebuild-when-it-matters is the documented
   pattern. Two more caller obligations replace deleted automation, both
   documented: a callback that replaces `x` or `args` and uses a metric
   built from them must rebuild the metric **in the same action** (the old
   `metric_valid` clearing is gone with the flags); and under multi-start,
   drawn lanes inherit the caller's initial instances, so a lane's *first*
   step runs under the initial metric — the callback corrects it from step
   two (the old per-lane `prepare` re-ran at the drawn point before the
   first step).
2. **Preconditioner changes are neither compared nor invalidated.**
   Staleness is safe by contract (it only changes the CG iteration path),
   so a refresh costs nothing beyond the refresh.
3. **Structure enforcement**: `_apply_action` guards the returned
   `metric`/`preconditioner` like it already guards `hyper` — treedef AND
   leaf dtype/weak-type must match the carried ones — with an error naming
   the field, rather than letting a raw carry mismatch surface downstream.
4. Ridge changes keep their existing detection (`_apply_action_state`).

### Renames

`LMSolveAction` → **`LMAction`**, `LMSolveContext` → **`LMContext`**. They
are the callback protocol's types; "solve" in the name was noise.
`LMSolveResult` keeps its name — it is specifically what `solve` returns.
`LMAction` keeps `stop`/`status` (spooky needs them). No aliases.

### Compilation identity

The solver's `_static_key` replaces the identity-hashed metric and
preconditioner components with `jax.tree_util.tree_structure(instance)`.
Linear-solver configs containing a preconditioner (`CG`, `GramCG`) are keyed
as (type, scalar knobs, treedef of the preconditioner). Equal-config
instances with fresh arrays therefore share one compiled loop; instances
with different static fields have different treedefs and key different
programs. Leaf *values* stay free to change because they enter the compiled
program as traced carry leaves; leaf *shapes/dtypes* retrace through the
ordinary jit input avals, not through the static key.

The keying rule is exactly "treedef-hash what is threaded, identity-hash
what is baked": an **explicit `ad_solver=CG(instance)`** keeps
identity-based keying for that instance, because its arrays enter the
tangent program as closed-over constants — treedef keying there would let
two solvers with different AD preconditioner values compare equal and
silently share one baked-in program.

New `test_compilation.py` pins:

- a fresh equal-config metric/preconditioner per solve compiles once (the
  old tests pinned the weaker reuse-the-same-object property);
- a callback-driven instance swap does not recompile;
- a changed static field does;
- **companion correctness pins** (the other half of the guarantee): two
  same-treedef different-valued metrics produce different solutions, and a
  callback-swapped preconditioner demonstrably changes the CG iteration
  path (tight `maxiter`, exact vs. identity), so cache sharing can never
  hide a static-read bug.

### Implicit AD

At the solution the converged instances ride in `result.lm_state`, which the
tangent rule already stop-gradients: their leaves are frozen inputs to the
implicit system, replacing `_frozen_ctx`'s prepare-at-solution. The
state-dependence of a callback-refreshed metric is not differentiated.

One deliberate contract change, stated plainly: `_frozen_ctx` used to
re-run `prepare` **at the returned solution**, whatever the forward refresh
policy had been; the new rule freezes the **carried** instance — the
geometry the solve actually converged under, which may be one refresh
behind the solution point. For a fixed metric (every real caller today) the
two are identical. An iterate-dependent metric that wants the old
at-solution semantics refreshes on every accepted step, making the carried
instance current at convergence. This is documented and pinned by a test
rather than silently absorbed.

Failed lanes: the tangent program must not read callback-mutated instances
from a failed solve (a swapped metric can be as invalid as a swapped ridge).
`_initial_ad_point` grows the pre-loop instances alongside the pre-loop
ridge it already carries, and `_ad_x_tangent` selects per-lane between the
result instances and the initial ones with the existing `_where_tree`
success mask — structure equality is already enforced, so the select is
well-posed.

AD-role preconditioning: `ad_solver=None` keeps today's inherit-the-forward
rule, now reading the **carried instance from `result.lm_state`** (the
callback-refreshed one at the solution). `ad_solver=CG(None, tol=...,
maxiter=...)` — `preconditioner=None` is newly legal in the AD role only —
also inherits the carried instance while pinning the AD tolerance and
budget (the kernels driver needs exactly this; with `ad_solver=None` the AD
budget falls back to CG defaults, which is tens of thousands of iterations
at that problem size). An explicit `ad_solver=CG(instance, ...)` uses the
given instance as-is (identity-keyed, see above). `preconditioner=None`
stays invalid in the forward role — opting out of forward preconditioning
remains the explicit `IdentityPreconditioner()`. The
`requires_positive_damping` exclusion is preserved in every AD form: an
inherited forward instance carrying that flag (e.g. `Padded`) falls back to
unpreconditioned, and an explicit one is rejected at construction, exactly
as today.

### Multi-start

Sequential and Python modes: drawn lanes start from `_cold_state` of the
caller's initial state, so they inherit the caller's initial instances and
never see another lane's callback mutations. Unchanged.

Parallel (vmapped) mode: **supported, no guard** — this resolves the open
question differently from the plan's lean. Instance leaves batch under
`vmap` like every other carried array (verified: an unbatched instance in
the initial carry with per-lane `lax.cond` rebuilds returns correctly
batched leaves), so per-lane mutation of leaf *values* is correct by
construction; lanes cannot diverge in structure, shape, or dtype — the same
rule as every other vmapped carry. The old design couldn't batch
identity-hashed closures, but the new one has nothing left to guard. The
real cost is documented instead: under `vmap` a `lax.cond` rebuild lowers
to a select that pays both branches every step — keep refresh logic behind
`lax.cond` for sequential drivers, and expect the rebuild cost per step if
you vmap it.

### `ridge_continuation` becomes `AnnealRidge`, a shipped convenience callback

The `ridge_continuation` factory and the `RidgeContinuation` name leave the
API. What ships instead is **`AnnealRidge`** — a frozen-dataclass callable
with today's full semantics and validation (`ridge_floor > 0`,
`0 < decrease < 1`, `grad_rtol > 0`, `stall_rtol` in `[0, 1)`), plus an
`init_state(dtype=None)` method replacing the factory's returned
`user_state0`:

```python
anneal = AnnealRidge(ridge_floor=1e-10)
result = solver.solve(x0, callback=anneal,
                      user_state=anneal.init_state(dtype),
                      gtol=1e-8, atol=1e-8)
```

Its **docstring documents the implementation** — the per-level reference in
`user_state`, the `+inf` fresh-level sentinel, the reset-on-advance, the
interplay with convergence suppression — so a driver that needs more than
annealing composes it inside its own callback rather than forking it:

```python
def driver_callback(ctx):
    action = anneal(ctx)
    advanced = action.lm_state.ridge < ctx.lm_state.ridge
    precond = jax.lax.cond(
        advanced,
        lambda: BlockEigenPreconditioner(
            build_families(ctx.x, ctx.args), PERMUTATION
        ),
        lambda: ctx.lm_state.preconditioner,
    )
    return dataclasses.replace(
        action,
        lm_state=dataclasses.replace(action.lm_state, preconditioner=precond),
    )
```

`lax.cond` branches trace once, so the `eigh` is paid only when the level
advances. The docs call out the anti-pattern explicitly: merging with
`jnp.where` over an unconditional rebuild pays the `eigh` every step — the
old kernels wrapper's actual behavior. The dtype handling keeps today's
contract: the stationarity comparison runs at the ridge dtype (the tracker
is cast in, results cast back to the tracker's dtype), and equal schedules
stay value-hashable so rebuilding one does not recompile the loop.

Callback hashing contract (documented): a callback is built once per
process — a module-level function, a frozen dataclass callable with
scalar-only fields, or a setup-scope closure the driver constructs once.
What recompiles is rebuilding a *fresh closure per solve call*. Closures
over large arrays (the kernels rebuild needs `K`, `K_tilde`, `F`, physics
constants — more than `ctx` carries) are fine at driver scope:
identity-hashed, one compile per run, and cross-run persistent-cache hits
are keyed on the traced HLO, which closure identity does not enter.

## Before/after: the external callers

### kernels `multicountry_growth_kernel.py` (CG path)

Before (~25 lines): `args = {"preconditioner": build_preconditioner_state(theta_0)}`,
a wrapper callback that calls the continuation, detects `advanced`, rebuilds
the state, `jnp.where`-merges it (paying the eigh every step), and threads it
back through `args`; an explicit `ad_solver=CG(BlockEigenPreconditioner(), ...)`
so the AD solve can find the state in `result.args`.

After: `multicountry_block_eigen_state` returns the family arrays; the
driver builds `BlockEigenPreconditioner(families(theta_0), permutation)`
once, passes it to `CG(...)`, and uses a driver-scope callback composing
the shipped `AnnealRidge` with the `lax.cond` rebuild closing over the
kernel matrices.
`ad_solver=CG(None, tol=lm_set.cg_tol, maxiter=lm_set.cg_maxiter)` inherits
the refreshed carried instance at the solution while keeping the pinned AD
budget, so the args-threading is deleted.

### spooky `mv2020_rbc_continuous.py`

`epoch_callback` (data re-draw + damping reset + custom stop on epoch
boundaries via `lax.cond`) keeps working verbatim modulo
`LMSolveAction` → `LMAction`. It touches neither metric nor preconditioner,
so no invalidation fires beyond the existing args-change rule.

### Full migration inventory

- `kernels/multicountry_growth_kernel.py` + `multicountry_growth_preconditioner.py`
  (above; the preconditioner module returns families instead of calling
  `block_eigen_state`).
- `kernels/open_economy_growth_kernel.py` (`ridge_continuation` →
  `AnnealRidge`).
- `kernels/extra/neoclassical_growth_kernel_recursive_adaptive.py`
  (`LMSolveAction` rename).
- `kernels/tests/test_python_models.py` → moves to `extra/tests/` (dead
  import + one dead-API test deleted).
- `kernels/trade_growth/` → moves to `extra/trade_growth/` wholesale; it
  targets a pre-2.7 API and is excluded from the sweep.
- `spooky/mv2020_rbc_continuous.py`, `spooky/mv2020_rbc_discrete.py`, and
  the annotated qmd (rename only; ported after kernels).

## Out of scope (settled elsewhere)

- **dtype policy**: one consistent dtype inferred from the problem; the
  float64-solve promotion knob stays out until a float32 test campaign
  motivates it (decision of 2026-07-25).
- No new `DampingSettings` grouping: `hyper` already is the settings group.

## Test plan (Stage 4)

- Compilation pins and companion correctness pins listed above.
- Callback swaps metric → ridge solver: that iteration cannot report
  `CONVERGED`, solver caches invalidated, Jacobian cache retained; metric
  solver: no suppression, caches invalidated. Preconditioner swap → no
  invalidation, demonstrably different CG path. Wrong-type/structure swap →
  loud `_apply_action` error naming the field.
- Block-eigen constructor numerics vs a dense reference.
- `AnnealRidge` reproduces today's `ridge_continuation` results on the
  existing fixtures (including the `stall_rtol` variant), and the
  composition recipe from its docstring works as written.
- Failed-lane AD with a callback-mutated metric uses the initial instances
  (extends the existing failed-lane ridge tests).
- The untouched-metric identity short-circuit: a callback returning
  `dataclasses.replace(ctx.lm_state, ridge=...)` every step adds no
  comparison ops for the metric subtree (jaxpr inspection).
- AD-contract pin: the tangent uses the carried (frozen) instances; the
  `requires_positive_damping` inherited-AD fallback still applies.
- Prune tests of deleted machinery.
- AD: rerun the reverse/forward tangent suites — the frozen-at-solution
  semantics replace `_frozen_ctx`.
