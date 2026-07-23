# Plan: `RidgeLevenbergMarquardt` (`ridge_lm.py`)

Add a second solver to `nlls_gram` that minimizes the **ridge objective**

```
F(x) = ||r(x, args, p)||^2 + ridge * q(x),      q(x) = ||L x||^2
```

over a JAX pytree `x`, where `L` is a user-supplied positive-semidefinite
penalty factor (`M0 = L'L`, allowed singular — e.g. `blockdiag(K,...,K, 0_d)`)
and `ridge` (the weight λ) is **traced state**, mutable from a `solve`
callback so a ridge continuation (homotopy λ ↓ 0) is a user-level recipe, not
solver machinery. The LM damping μ is a plain Euclidean trust-region
parameter, fully decoupled from λ.

Equivalent view (used everywhere in the implementation): standard Euclidean
LM on the **augmented residual**

```
R(x) = [ r(x) ; sqrt(ridge) * L x ],     A = [ J ; sqrt(ridge) * L ]
step: (A'A + damping*I) δ = -(J'r + ridge*L'(L x)) = -g
```

The augmented rows are affine in `x`, so geodesic acceleration, accept/reject,
and the implicit-AD rule all reduce to the standard LM formulas with `A` in
place of `J` and second-derivative contributions from the penalty identically
zero.

Target selection semantics (why this solver exists): under
`ker J ∩ ker L = {0}` at the solution — plus the standard local regularity
of an isolated constrained minimizer (the claims below are local; the affine
case is worked out exactly in the companion paper appendix) — each fixed-λ
problem has an isolated minimizer `x_λ`, and `x_λ → x† = argmin q s.t. r = 0`
with `O(λ)` error — the minimum-seminorm (min-RKHS-norm) interpolant. No metric solves, no epsilon
shift on the zero-padded scalars, no whitening.

The design puts the selection in the OBJECTIVE (classical Tikhonov
regularization of nonlinear problems: convergence of ridge minimizers to a
minimum-seminorm solution as λ ↓ 0 [EKN89; EHN96 Ch. 10]; the seminorm
penalty `||Lx||^2` and the `ker J ∩ ker L = {0}` well-posedness condition go
back to [Elden82]) rather than in the algorithm's implicit bias. That choice
is deliberate: plain Gauss–Newton on an underdetermined system does not by
itself produce the minimal-norm solution ([CKB12], whose fix is an explicit
null-space correction, relaxed further in [PR22]), and modern local GN
convergence theory guarantees convergence to *some* solution on the manifold
while characterizing WHICH one only when the residual is affine ([IS26]).
With the ridge objective, every inner problem is a well-posed NLLS that any
solver can attack with textbook guarantees, and the annealed one-pass limit
of the continuation is the iteratively regularized Gauss–Newton method
([Bak92; BNS97; KNS08]). For kernel instantiations, each inner LM step is a
kernel ridge regression of the relinearized equations, as in [CHOS21].

## Ground rules

- **Copy `gram_lm.py` → `ridge_lm.py`, then rip out and modify.** Keep the
  loop/callback/multi-start/AD skeleton and amortization tricks; delete the
  metric/whitening/dual machinery. Considerably shorter than gram_lm
  (~1,200–1,600 lines expected).
- **Duck-type compatibility.** Same init/update/solve protocol, same field
  names where semantics carry over, same callback contract
  (`LMSolveContext -> LMSolveAction | None`), same `LMSolveResult`. User code
  written against `LevenbergMarquardt` should port by changing the
  constructor call and (if it inspects state) reading the new fields.
- Shared pieces move to a new `src/nlls_gram/utility.py`; `gram_lm.py`
  imports them and **re-exports the public names** so existing imports keep
  working. No behavior change to `LevenbergMarquardt`; its tests must pass
  untouched.
- Public API gets docstrings (published package); internals get none.
- All numerics float-dtype-generic (float32/float64), following the existing
  dtype-handling patterns (`_cast_hyper`, `linear_solve_dtype` promotion).

## 1. New module `utility.py` (mechanical move, no logic changes)

Move from `gram_lm.py`, re-export from `gram_lm.py` for compatibility:

- `LMStatus`, `LMHyperparams`, `_cast_hyper`, `_damping_floor`
- `LMSolveAction`, `LMSolveContext`, `LMSolveResult`, `MultiStartInfo`,
  `MultiStart`, `DrawNNXModule`
- `canonicalize_residual`
- hashing/identity helpers: `_typed_key`, `_IdentityKey`, `_IdentityCallable`,
  `_static_key_component`, `_hashable_hook`
- tree helpers: `_tree_changed`, `_zero_tangent_leaf`,
  `_broadcast_leading_condition`, `_where_tree`, `_mask_tangent_tree`
- history helpers: `_history_buffer`, `_init_history`, `_record_history`,
  `_finalize_history`
- loop helpers: `_solve_loop_impl`, `_accept_converged`,
  `_accept_converged_or_max_steps`, `_ranking_loss`, `_attempt_success`,
  `_type_spec`, `_check_drawn_types`, `_cold_lm_state`

**The shared-loop protocol (audit result — the coupling is wider than
`update`/`_converged`).** `_solve_loop_impl` and the multi-start drivers
touch: `solver.has_aux`, `solver._residual_and_aux`, `solver._initial_info`,
`solver.update`, `solver._apply_action`, `solver._converged`, and they
directly replace `lm_state.damping` and `.hyper` with dtype-cast values.
The move list must also include the free-function multi-start
sequential/parallel implementations and their stable jitted wrappers at the
bottom of `gram_lm.py` (below line ~3830), with `_solve_impl`,
`_multi_start_impl`, and `_multi_start_python` remaining thin methods that
delegate into the shared drivers. Parameterize the loop on an explicit
informal solver protocol instead of ad-hoc attribute reads:

- `_cast_state(lm_state, dtype)` — replaces the loop's direct
  damping/hyper casts; the ridge version also casts `ridge` (and `qr_ridge`)
  to the residual dtype with stable scalar shapes.
- `_initial_info`, `_converged`, `_residual_and_aux`, `update` — as today.
- `_apply_action(action, ...)` — solver-specific: the ridge version must
  treat a callback that changes `lm_state.ridge` as a PROBLEM CHANGE, i.e.
  suppress the convergence check for that iteration (the diagnostics were
  computed at the old ridge) and invalidate the ridge-dependent caches
  (`qr_valid`, and `penalty_valid` if a factory ships). The existing
  `_apply_action` only knows about `x`/`args` changes — do not inherit that
  blind spot. Callback-provided ridge contract: scalar, finite, positive.
- `_ranking_objective(result)` — multi-start ranking hook. The existing
  `_ranking_loss` recomputes `sum(residual**2)` whenever a callback exists,
  which would silently drop the penalty term for every continuation run.
  Gram implementation returns `||r||^2` (unchanged behavior); ridge returns
  `||r||^2 + ridge * q(x)` at the lane's own final ridge (with a documented
  comparability caveat: under a shared continuation schedule the lanes'
  ridges agree in practice).
- `_cold_state(lm_state)` — multi-start lane reset. Match the EXISTING
  semantics: invalidate caches and zero factory states, but PRESERVE the
  caller-supplied `damping`, `hyper`, and (new) `ridge` from the initial
  state — do NOT reset to constructor defaults (parallel lane 0 uses the
  cold state while sequential attempt 0 uses the original; constructor
  resets would make them disagree, and `ridge=None` cannot be resolved
  without the residual dtype anyway).

`LMState`/`LMInfo` do NOT move — each solver defines its own (`gram_lm` keeps
its current definitions; `ridge_lm` defines `RidgeLMState`/`RidgeLMInfo`).

## 2. New module `penalties.py` — the `RidgePenalty` type

```python
@dataclass(frozen=True)
class RidgePenalty:
    """Positive-semidefinite penalty q(x) = ||L x||^2 given through callbacks.

    All callbacks act on the flattened parameter vector. `sqrt_apply` may
    return a vector of any length k (the number of penalty rows); `M0 = L'L`
    is never formed unless `add_scaled` is omitted.

    - sqrt_apply(x):            L x                      (required)
    - sqrt_transpose_apply(y):  L' y                     (required)
    - quadratic(x):             x' L'L x  (default: ||sqrt_apply(x)||^2)
    - add_scaled(H, c):         H + c * L'L for a dense (p, p) H
                                (optional; default materializes via
                                sqrt_apply on an identity basis — provide it
                                for anything large)
    - num_rows:                 static int k, required by the dense/qr paths
                                to size the augmented system
    """
```

Constructors (mirror `metrics.py` patterns, reuse its packed-head/unpack
tricks and `quasiseparable`):

- `repeated_dense_penalty(K, *, repeats, zero_pad_size)` — the workhorse.
  `q(x) = sum_j alpha_j' K alpha_j` over `repeats` head blocks, scalars in the
  tail unpenalized. Factor `K = C C'` (one `jnp.linalg.cholesky`; document
  that the caller may add jitter to `K` if numerically semidefinite);
  `sqrt_apply` = batched `C' @ alpha_j` over blocks (the existing
  `packed_head` batching), zero-pad tail contributes no rows
  (`num_rows = repeats * N`). `add_scaled` scatter-adds `c*K` into the
  head diagonal blocks of `H` (no blockdiag materialization).
- `repeated_state_space_penalty(t, h, Pinf, transition, *, repeats,
  zero_pad_size, parallel=None)` — O(N) Matérn version. AUDIT RESULT: with
  `K = C C'`, `quasiseparable._cholesky_transpose_matvec` supplies
  `sqrt_apply(x) = C'x`, but the companion `sqrt_transpose_apply(y) = C y`
  does NOT exist yet — `_forward_substitution`/`_backward_substitution` are
  SOLVES with `C`/`C'`, not factor matvecs. Implement a new
  `quasiseparable._cholesky_matvec` scan (the direct companion of the
  existing transpose matvec) as part of this step. `add_scaled` is omitted
  here (dense assembly of a state-space penalty defeats the point); the
  cholesky linear solver then requires a penalty with `add_scaled` or falls
  back to materialization — document that state-space penalties pair with
  `linear_solver="lsmr"`.
- `penalty_from_factor(L)` — generic dense factor (k, p).
- `identity_penalty(size)` — `L = I`, for min-Euclidean-norm problems and
  tests.

**`PenaltyFactory` is DEFERRED to phase 2, and its contract must be
narrower than `MetricFactory`'s.** Copying `prepare(x, args, p, aux)` would
be mathematically invalid here: the gradient `J'r + λL'Lx` and the implicit
AD rule treat `L` as constant, so an `x`-dependent factor would silently
drop the `d/dx [L(x)'L(x)x]` terms and a `p`-dependent factor would drop
`d(L'Lx)/dp` from the AD right-hand side. Phase 2 may ship a factory whose
prepared data is restricted to differentiation-inert problem data (e.g.
rebuilt from `args` under `stop_gradient`), with that restriction validated
and documented. Phase 1: constructor reserves the `penalty_factory` kwarg
and raises `NotImplementedError`.

## 3. `ridge_lm.py` public API

```python
class RidgeLevenbergMarquardt:
    def __init__(
        self,
        residual_fn,                 # (x) | (x, args) | (x, args, p), has_aux supported
        *,
        penalty,                     # RidgePenalty (required)
        penalty_factory=None,        # exclusive with penalty if kept in phase 1
        ridge=None,                  # initial λ; None -> sqrt(finfo(dtype).eps)
                                     #   resolved at init() from the residual dtype
                                     #   (float64 ≈ 1.5e-8, float32 ≈ 3.5e-4).
                                     #   STRICTLY POSITIVE by contract: ridge = 0 is
                                     #   unsupported everywhere (constructor validates,
                                     #   callbacks must keep it positive, continuation
                                     #   floors are positive) — so J'J + ridge*L'L is
                                     #   always PD under ker J ∩ ker L = {0}
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        min_damping=None,            # same underflow-floor semantics as gram_lm
        max_damping=None,
        linear_solver="auto",        # "auto" (= "cholesky") | "cholesky" | "qr" | "lsmr"
        jacobian_mode="auto",        # "auto" | "fwd" | "rev"; auto: JVP columns when
                                     #   p <= n_resid, else VJP rows — vmap over the
                                     #   SMALLER side, same rule as gram_lm.py:1636
                                     #   (TRUE residual only; penalty rows never use AD)
        iterative_tol=0.0,           # lsmr stopping, same mapping as gram_lm
        iterative_atol=0.0,
        iterative_maxiter=8,
        lsmr_preconditioner=None,    # WhitenedPreconditioner, REQUIRED for "lsmr";
                                     #   identity_right_preconditioner() to opt out
        ad_solver="auto",            # "auto" | "cholesky" | "normal_cg"
        ad_solver_tol=None, ad_solver_atol=0.0, ad_solver_maxiter=None,
        ad_solver_preconditioner=None,  # optional SPD preconditioner for the
                                     #   normal_cg AD solve — kept from gram_lm:
                                     #   unpreconditioned CG on J'J + λM0 degrades
                                     #   as λ shrinks, so the hook stays
        linear_solve_dtype=None,     # None | jnp.float64, dense paths only
        has_aux=False,
        cache_jacobian=True,         # dense paths INCLUDING "qr" (unlike gram_lm,
                                     #   which disables it for qr — here the qr_R
                                     #   cache is only useful with cached resid/Jt,
                                     #   so the two cache under one validity flag)
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
    )
```

Removed relative to `LevenbergMarquardt` (delete the code paths outright):
`metric`, `metric_factory`, `metric_solve_dtype`, `dual_preconditioner`,
`preconditioner_factory`, `normal_preconditioner`, `recycle` and all
deflated-CG machinery, solvers `gram_cholesky`/`normal_cholesky`(as names)/
`gram_cg`/`normal_cg`/`augmented_qr`, AD methods `direct`/`svd`/`qr`/
`augmented_qr`/`gram_cg`/`regularized_normal_cg`.

Constructor validation mirrors gram_lm's style: unknown solver names,
positivity checks, `lsmr_preconditioner` required iff `linear_solver="lsmr"`
(error message tells the user to pass `identity_right_preconditioner()`),
`jacobian_mode != "auto"` rejected for `"lsmr"` unless a dense AD method
consumes it, `linear_solve_dtype` restricted to dense paths + x64 check.
Value-based `__eq__`/`__hash__` static key over constructor args (copy the
pattern) so equal-config solvers share compiled loops.

### State, info, and hyper

```python
@register_dataclass
@dataclass(frozen=True)
class RidgeLMState:
    damping: jax.Array                 # μ
    ridge: jax.Array                   # λ (traced; callbacks may replace it)
    resid: jax.Array | None = None     # cache_jacobian: TRUE residual at x
    Jt: jax.Array | None = None        # cache_jacobian: J' of TRUE residual
    jacobian_valid: jax.Array | None = None
    aux: Any = None
    hyper: LMHyperparams | None = None
    qr_R: jax.Array | None = None      # qr path: R factor of [J; sqrt(λ)L], (p, p)
    qr_valid: jax.Array | None = None  # qr_R current for (x, ridge)
    qr_ridge: jax.Array | None = None  # λ the cached qr_R was built with
    penalty_state: Any = None          # penalty_factory prepared state
    penalty_valid: jax.Array | None = None
```

`LMHyperparams` reused unchanged from `utility.py` (ridge is state, not
hyper — it parallels `damping`, and `damping` also lives at state top level).

```python
@register_dataclass
@dataclass(frozen=True)
class RidgeLMInfo:
    loss: jax.Array            # objective ||r||^2 + λ q at the retained iterate
    loss_old: jax.Array        # objective at the pre-step x
    loss_candidate: jax.Array  # objective at the trial point
    resid_loss: jax.Array      # ||r||^2 at the retained iterate  (NEW)
    penalty_value: jax.Array   # q(x) at the retained iterate     (NEW)
    ridge: jax.Array           # λ used this step                 (NEW)
    accepted: jax.Array
    damping: jax.Array
    damping_factor: jax.Array
    used_geodesic: jax.Array
    acceleration_ratio: jax.Array
    grad_norm: jax.Array       # ||J'r + λ M0 x|| — ridge stationarity, NOT ||J'r||
    step_norm: jax.Array       # Euclidean
    aux: Any = None
```

Field names shared with `LMInfo` keep their positions/meanings so
callback/user code reading `info.loss`, `info.accepted`, `info.damping`
ports unchanged; `loss` is the objective actually being minimized (document
prominently — it includes the penalty).

### `update(x, lm_state, args=None, p=None)`

Same skeleton as gram_lm's `update`:

1. Flatten `x`; build TRUE-residual closures (`has_aux` handling identical).
2. Dense paths: materialize `Jt` per `jacobian_mode` with the
   `cache_jacobian`/`jacobian_valid` `lax.cond` reuse pattern (unchanged).
   lsmr: `jax.linearize` for matvec closures (unchanged).
3. Read `hyper` (or constructor fallback), floor `damping`; read
   `ridge = lm_state.ridge` (assert present; `init()` populates it).
4. Resolve penalty callbacks (fixed `penalty` or factory build with the
   `penalty_valid` reuse-on-reject `lax.cond` — copy the metric_factory
   pattern verbatim).
5. Gradient direction `g = J'r + ridge * sqrt_transpose_apply(sqrt_apply(theta))`
   (the HALF-gradient: `∇F = 2g`; the factor cancels in the LM equations but
   docstrings must not call `g` "the gradient" unqualified);
   `grad_norm = ||g||`.
6. `solve_step(g_vec)` returning `δ = -(A'A + damping I)^{-1} g_vec` per
   `linear_solver` (below). Velocity: `solve_step(g)`. Geodesic ([TS12]):
   `f_vv` via jvp-of-jvp of the TRUE residual only (penalty rows are affine —
   zero second derivative by construction, so never compute them);
   `acceleration = solve_step(J' f_vv)`. Acceptance-ratio norms are
   **Euclidean** (no metric norm anymore).
7. Objective at trial points: `||r(x+δ)||^2 + ridge * quadratic(x+δ)`.
   Accept iff finite and strictly below the pre-step objective (computed with
   the SAME `ridge` — a callback changing λ between steps changes the
   objective only for subsequent steps, which keeps per-step monotonicity
   well-defined).
8. Damping update, cache threading (`~improved` validity flags), build
   `RidgeLMState` + `RidgeLMInfo`. `ridge` passes through unchanged —
   only `init()` and callbacks set it.

`atol`/`gtol`/`xtol` in `_converged` — **atol is a conjunctive filter, never
a sufficient condition** (review finding: a pure-residual atol would
terminate at step 0 from any interpolating start with `Lx != 0`, before any
seminorm minimization happens — and the kernels/spooky drivers use exactly
residual-only atol today). Semantics:

- Convergence fires when (`gtol` on `grad_norm` — ridge stationarity — or
  `xtol` on the accepted Euclidean step norm) **AND** (`atol == 0` or
  `sqrt(resid_loss) <= atol`). atol alone never stops the solve.
- `solve` validates `atol > 0` requires `gtol > 0` or `xtol > 0` (loud
  error, message explains why residual-only stopping is wrong for a ridge
  objective).
- Document: gtol/xtol mean "done with the current fixed-λ problem"; atol
  additionally demands "and the model equations are actually solved" — the
  ridgeless-endgame check. In a continuation run the callback keeps lowering
  λ, so intermediate stationarity with a large residual correctly does not
  stop the loop.

### Linear solvers

All three solve `(J'J + λ L'L + μ I) δ = -g_vec`, share one factorization
between velocity and acceleration within an update, and must agree to
tolerance (tested).

**`"cholesky"` (= `"auto"`), the default.** Dense normal equations:
`H = Jt @ Jt.T` (promoted to `linear_solve_dtype` BEFORE the product, same
wide-pipeline recipe and comment as gram_lm's cholesky branches),
`H = penalty.add_scaled(H, ridge)`, add `damping` to the diagonal,
`cho_factor` once per update, `cho_solve` per RHS. Amortization: `resid`/`Jt`
reuse on rejected steps via the existing cache; `H` is reassembled each
update from the cached `Jt` (matches gram_lm, which also reassembles its Gram
matrix per update — assembly is one GEMM, factorization O(p³/3); acceptable,
and it makes λ-changes-via-callback automatically correct with no extra
invalidation logic). If `penalty.add_scaled` is missing, materialize `M0`
once at trace time from `sqrt_apply` columns only when `p` is small; error
with a clear message otherwise.

**`"qr"`.** MINPACK-structured ([More78]'s damping-row QR update), stable at
small `λ`/`μ` where forming `H` squares the condition number:

- At a fresh `(x, ridge)`: stack `A = [Jt.T; sqrt(ridge) * L_rows]` (the
  penalty rows materialized via `sqrt_apply` on the identity — requires
  `penalty.num_rows`; for the repeated-dense penalty this is
  `sqrt(ridge) * blockdiag(C', ..., C')` assembled by the penalty itself via
  an optional `sqrt_rows()` callback returning the dense (k, p) factor —
  add that callback to `RidgePenalty`, default builds it from `sqrt_apply`),
  and compute `R = qr([A], mode="r")` (p×p). Cache `(qr_R, qr_ridge)` in
  state with `qr_valid = jacobian_valid`-style reuse; a reject reuses `R`,
  and validity additionally requires `lm_state.ridge == qr_ridge` (callback
  λ-changes invalidate).
- Per update (every `damping`): `R_mu = qr([R; sqrt(damping) I], mode="r")`.
  Cost honesty: a generic (2p, p) QR is still cubic; MINPACK's actual
  `qrsolv` eliminates the diagonal damping rows with Givens rotations in
  O(p²) — implement the plain stacked QR first and note the Givens
  elimination as a follow-up optimization. Shape caveat: `R` is p×p only
  when `n + k >= p`; when the stack is wider than tall, `R` is rectangular
  and the damping rows are what make the final system full rank — size the
  buffers accordingly.
- Cache invalidation (beyond ridge equality): a callback that replaces `x`
  or `args` must clear `qr_valid` (and `penalty_valid`) exactly as it clears
  `jacobian_valid` today — extend the solver's `_apply_action` accordingly.
- Solve via **corrected semi-normal equations** ([Bjorck87]; the
  one-refinement-pass prescription is [Bjorck96] Sec. 6.6.5): triangular solves
  `R_mu' R_mu δ = -g_vec`, then ONE fixed iterative-refinement pass
  (`s = -g_vec - (J'(J δ) + ridge*L'(L δ) + damping*δ)` using matvecs;
  `δ += R_mu^{-1} R_mu^{-T} s`). Deterministic cost, no stored Q. Accuracy
  honesty: one correction is the conventional CSNE prescription but does not
  unconditionally reach full-QR accuracy on ill-conditioned problems; the μ
  and λ shifts bound the conditioning in practice, and the documented
  fallback (full stacked QR per update) stays local to `solve_step` if tests
  demand it.

**`"lsmr"`.** Matrix-free ([FS11]; right preconditioning per [Bjorck96]
Ch. 7 — a right preconditioner leaves the least-squares objective unchanged,
which is why it is the only correct side here), mirrors gram_lm's lsmr
branch with the metric stripped and `B → A`:

- Operator: `A_matvec(v) = concat([jvp(v), sqrt(ridge) * sqrt_apply(v)])`,
  `At_matvec(w) = vjp(w[:m]) + sqrt(ridge) * sqrt_transpose_apply(w[m:])`.
- Right preconditioner `R^{-1}` from the REQUIRED `lsmr_preconditioner`
  (reuse the `WhitenedPreconditioner` type: `solve(v, damping)`,
  `solve_transpose(w, damping)`); ship `identity_right_preconditioner()`
  in `preconditioners.py`. Same argument as gram_lm: the augmented damping
  row is posed in the UNpreconditioned variable so every `damping > 0`
  subproblem is exactly the I-damped problem for ANY `R` — copy that
  construction (`A_aug(z) = [A(Rinv z); sqrt(damping) * Rinv z]`,
  `b_aug = [0; c / sqrt(damping)]` targeting arbitrary RHS `c`), including
  the `custom_linear_solve` wrapper around `N = A'A + damping I` so the
  forward solve is AD-differentiable, and the `4 * min(m + k, p)` cap that
  applies only when `iterative_maxiter=None` (the constructor default is 8,
  exactly as in gram_lm — do not advertise the cap as the default).
- Tolerance mapping `iterative_tol -> atol`, `iterative_atol -> btol`
  unchanged (`lsmr.py` untouched).

### `solve(...)`

Signature and semantics copied verbatim (all of: `p`, `lm_state`,
`max_steps`, `max_steps_is_success`, `atol/gtol/xtol`, `callback`,
`user_state`, `save_steps`, `multi_start`, `jit`), driving the shared
`_solve_loop_impl`. Differences:

- `init(x0, args, p=None)` populates `ridge` (constructor value or the
  dtype default) alongside `damping`, plus the caches the configuration
  needs. `solve` with `lm_state=None` calls `init` UNCONDITIONALLY (unlike
  gram_lm's conditional shortcut): resolving `ridge=None` needs the residual
  dtype, so there is no valid minimal-state fast path. A caller-supplied
  `lm_state` must already carry a positive `ridge`; validated loudly.
- The **ridge continuation recipe is a documented callback**, not a solver
  feature. A `lax.while_loop` carry cannot grow `user_state` from `None`
  mid-loop, so the helper is a FACTORY returning both the callback and its
  fixed-shape initial tracking state:

```python
def ridge_continuation(*, decrease=0.1, ridge_floor, grad_rtol=1e-3, dtype=None):
    """Build (callback, user_state0) implementing ridge continuation.

    The callback multiplies lm_state.ridge by `decrease` whenever the inner
    problem is approximately stationary (grad_norm below grad_rtol relative
    to its reference value at the current ridge level), never below
    ridge_floor. ridge_floor is REQUIRED and strictly positive (ridge = 0 is
    out of contract). The callback returns an LMSolveAction replacing
    lm_state AND user_state (the per-level reference grad_norm lives in
    user_state as fixed-shape scalars).

    Usage: cb, us0 = ridge_continuation(ridge_floor=1e-10)
           solver.solve(x0, callback=cb, user_state=us0, gtol=..., atol=...)
    """
```

  `solve` needs no new parameters — the user's hunch holds. A ridge change
  by the callback is a problem change (see the `_apply_action` contract in
  Sec. 1): convergence is suppressed for that iteration and the qr/penalty
  caches invalidate. Tests exercise both fixed-λ and continuation runs.

  Theory anchors to cite in the helper's docstring and docs page: the
  solved-out continuation path converges to the minimum-seminorm solution by
  nonlinear Tikhonov theory [EKN89; EHN96], while annealing λ per accepted
  step instead of per solved level is the iteratively regularized
  Gauss–Newton method [Bak92; BNS97; KNS08] — whose theory wants a monotone,
  boundedly geometric schedule; the helper's multiply-on-stationarity rule
  satisfies that along accepted levels.

### Implicit AD (`ad_solver`)

Same `custom_jvp`-on-`solve` skeleton (including multi-start variant, the
zero-tangent failure policy, `max_steps_is_success` handling, and
stop-gradient initial-point fallback for failed lanes). The tangent rule is
**Gauss–Newton implicit differentiation of the ridge stationarity**
`J'r + λ M0 x = 0`. Exact differentiation carries TWO extra terms beyond
GN — `sum_i r_i ∇²_x r_i` inside the left-hand matrix and `(∂J'/∂p)[p_dot] r`
on the right — and the rule drops BOTH:

```
(J'J + λ M0) x_dot = -J' (∂r/∂p) p_dot        # λ frozen at the returned state's ridge,
                                              # stop_gradient'd: λ is inert data
```

NO damping in the AD matrix. Both dropped terms vanish when the converged
TRUE residual is exactly 0 (the ridgeless endgame — our regime); at moderate
λ the perturbation is first-order in `||r||`, which translates to a
first-order tangent error only under conditioning assumptions — state the
contract that way, not as an unconditional `O(||r||)` relative bound.
Because ridge = 0 is out of contract, `λ > 0` always holds and the matrix is
PD under `ker J ∩ ker L = {0}` at every reachable state — no
rank-deficiency, no min-norm-tangent machinery, which is why the gram_lm AD
menu collapses to two methods:

- `"cholesky"`: assemble `J'J + λ M0` (wide under `linear_solve_dtype`),
  `cho_solve`. Resolution of `"auto"` for dense forwards.
- `"normal_cg"`: matrix-free CG on the same operator (matvec =
  `vjp(jvp(v)) + λ L'(L v)`), with the OPTIONAL `ad_solver_preconditioner`
  hook retained from gram_lm: PD does not mean well-conditioned, and
  unpreconditioned CG iteration counts grow as λ shrinks. Resolution of
  `"auto"` for an lsmr forward. `ad_solver_tol/atol/maxiter` as stopping
  controls, with the same "maxiter required when both tolerances are zero"
  validation.

Threading (skeleton change vs gram_lm): `_ad_result_tangent` reads λ from
`result.lm_state.ridge` (available on every `LMSolveResult`), stop-gradient
applied — so the multi-start winner's own final ridge is used. Failed lanes
evaluate their (masked) tangent program at the stop-gradient INITIAL ridge
from the pre-loop state, never at a possibly-NaN callback-produced value.
Document that λ-inertness means the derivative deliberately ignores both the
continuation schedule's and the multi-start winner-selection's dependence on
`p`. `result.aux` tangents, `result.p` identity pass-through, history
buffers differentiation-inert: copy the existing treatment.

## 4. Exports, docs, and README

- `__init__.py`: export `RidgeLevenbergMarquardt`, `RidgeLMState`,
  `RidgeLMInfo`, `RidgePenalty`, `PenaltyFactory` (if shipped),
  `repeated_dense_penalty`, `repeated_state_space_penalty`,
  `penalty_from_factor`, `identity_penalty`,
  `identity_right_preconditioner`, `ridge_continuation`. Everything currently
  exported stays exported (utility move is invisible).
- `mkdocs.yml` + `docs/ridge_lm.md` — this page must EXPLAIN THE ALGORITHM,
  not just the API. Required content, in order:
  1. Problem statement: minimum-seminorm solution of an underdetermined
     nonlinear least-squares system, `x† = argmin ||Lx||^2 s.t. r(x) = 0`,
     with the `ker J ∩ ker L = {0}` identification condition [Elden82].
  2. The ridge family `||r||^2 + λ||Lx||^2` and the selection theorem
     sketch: isolated minimizers for fixed λ, `x_λ = x† + O(λ)` as λ ↓ 0
     [EKN89; EHN96 Ch. 10]. Why the objective carries the selection: plain
     GN on the underdetermined system converges to *some* solution and does
     not select the minimal-norm one without explicit correction
     [CKB12; PR22; IS26].
  3. The algorithm box: stock LM on the augmented residual
     `[r; sqrt(λ) L x]` with Euclidean damping μ; the λ/μ decoupling (μ =
     trust region, standard accept/reject [Marq63; More78; NW06 Ch. 10];
     λ = selection weight, annealed monotonically); geodesic acceleration
     [TS12]; the continuation recipe and its one-pass IRGNM limit
     [Bak92; BNS97; KNS08].
  4. Kernel instantiation: `repeated_dense_penalty` /
     `repeated_state_space_penalty`, each inner step = kernel ridge
     regression of the linearized equations [CHOS21], scalars unpenalized.
  5. Solver selection table (cholesky/qr/lsmr; cite [More78] and
     [Bjorck87; Bjorck96] for the qr path, [FS11] for lsmr) and the AD
     contract (GN-implicit, exact in the interpolating limit).
  6. Migration snippet from `LevenbergMarquardt` (Sec. 6 below) and a
     References list (Sec. 8 below, formatted).
- **README restructure.** The README leads with TWO main examples of equal
  weight:
  1. `RidgeLevenbergMarquardt` — min-seminorm/ridgeless interpolation via
     ridge continuation (the kernel example: `repeated_dense_penalty` +
     `ridge_continuation`), with a two-sentence algorithm summary and
     pointers to [EHN96; KNS08; CKB12].
  2. `LevenbergMarquardt` — general nonlinear least squares with the
     standard identity-damped LM as the presented configuration. DOWNPLAY
     the metric: the example passes no `metric` argument; the metric-damped
     variant moves to a short "advanced: metric damping" note linking to its
     docs page rather than appearing in the headline example.
  Forward-looking note for the implementer (do not put in the README): if
  `RidgeLevenbergMarquardt` proves out in the kernels/spooky repos, the
  metric machinery in `gram_lm.py` (metric/metric_factory, whitening, the
  gram dual solvers, metric_solve_dtype) is a candidate for deprecation and
  removal in a later major version. Keep the new code free of any dependency
  on metric internals, and keep the `utility.py` split clean, so that
  rip-out stays a local change to `gram_lm.py`.
- Module docstring of `ridge_lm.py` mirrors the package docstring style and
  includes the short-form citations for the objective, the continuation, and
  the qr/lsmr numerics.

## 5. Tests (`tests/`)

New files, mirroring the existing suites' granularity:

- `test_ridge_lm.py`: linear-Gaussian analytic problem where
  `x† = argmin q s.t. Jx = b` is known in closed form — fixed small λ lands
  within `O(λ)`; continuation callback reaches `x†` tighter than any single
  moderate λ; scalars (zero-pad block) are NOT shrunk (regression test for
  the no-epsilon-on-scalars property); accept/reject monotonicity of the
  objective; `grad_norm` is the ridge stationarity.
- `test_ridge_solvers.py`: cholesky/qr/lsmr step agreement on the same
  `(x, λ, μ)` to solver tolerance, float32 and float64; qr cache reuse on
  reject (count factorizations via a wrapped `qr`); λ-change invalidates the
  qr cache; lsmr with identity vs a nontrivial right preconditioner agrees.
- `test_ridge_penalties.py`: `repeated_dense_penalty` vs
  `repeated_state_space_penalty` agreement (Matérn on a 1-D grid, dense K
  from the same kernel); `add_scaled` vs materialized `M0`; `num_rows`/
  `sqrt_rows` consistency.
- `test_ridge_ad.py`: `jax.grad` of a function of `solve(...).x` w.r.t. `p`
  vs central finite differences at small λ (interpolating regime, tight),
  and at moderate λ (documented-bias regime, loose tolerance on purpose,
  asserting the *direction* of the contract, not equality); failed-status
  zero tangents; `ad_solver="normal_cg"` vs `"cholesky"` agreement.
- `test_ridge_solve_features.py`: callback replace of `lm_state.ridge`
  mid-solve; `save_steps` histories; `multi_start` sequential and parallel
  (cold lanes reset ridge and damping); `has_aux`; `jit=False` parity.
- `test_gram_lm.py` and the rest of the existing suite: run untouched (the
  utility move must be invisible). Add one cross-solver test: metric-LM
  (`repeated_shifted_dense_metric`, small ε) and ridge-LM (continuation)
  agree on the affine problem's `x†` within their respective ε/λ biases.

Run: `uv run pytest` in `nlls_gram` (x64 enabled per existing conftest
conventions — check `tests/test_float64_subprocess.py` for the pattern).

## 6. Migration snippet (kernels / spooky repos)

```python
# before (metric LM)
metric = repeated_shifted_dense_metric(K, repeats=3, zero_pad_size=d, epsilon=1e-7)
solver = LevenbergMarquardt(residual_fn, metric=metric,
                            linear_solve_dtype=jnp.float64,
                            metric_solve_dtype=jnp.float64, ...)

# after (ridge LM)
penalty = repeated_dense_penalty(K, repeats=3, zero_pad_size=d)   # no epsilon
solver = RidgeLevenbergMarquardt(residual_fn, penalty=penalty,
                                 ridge=1e-8,                       # or None for dtype default
                                 linear_solve_dtype=jnp.float64, ...)
cb, us0 = ridge_continuation(ridge_floor=1e-10)                    # optional homotopy
result = solver.solve(theta_0, max_steps=..., gtol=..., atol=residual_atol,
                      callback=cb, user_state=us0)
```

Consumer-migration audit (real call sites, checked during review):

- `spooky` reads and ranks by `result.info.loss` in places that mean
  equation error (e.g. `notebooks/growth_recursive_advanced.py`) — those
  sites must switch to `resid_loss`, since ridge `info.loss` includes the
  penalty.
- Residual-only `atol` stopping (both repos' drivers) must add a `gtol` (or
  `xtol`) per the conjunctive atol contract above; the solve-time validation
  makes this loud, not silent.
- `metric_solve_dtype=jnp.float64` has NO ridge analog; penalty callbacks
  follow their input dtypes. If a float32 program needs a float64 penalty
  pipeline, that is a phase-2 `penalty_compute_dtype` knob — note it in the
  docs rather than silently dropping the capability.
- LSMR users must now pass `lsmr_preconditioner=` explicitly
  (`identity_right_preconditioner()` at minimum) — a deliberate breaking
  difference from gram_lm's optional `whitened_preconditioner=`; both the
  rename and the requiredness go in the migration notes.
- The shared `LMSolveContext`/`LMSolveResult` docstrings currently name
  `LMState`/`LMInfo` — generalize the wording when they move to
  `utility.py` so they describe both solvers' state/info types.
- `linear_solve_dtype` policy for ridge assembly: promote BEFORE forming
  `ridge * K` contributions (a float32-rounded `ridge*K` added into a
  float64 `H` defeats the promotion).

## 7. Implementation order

1. `utility.py` extraction + gram_lm re-exports; full existing suite green.
2. `penalties.py` (`RidgePenalty`, dense constructors, `identity_penalty`)
   + `test_ridge_penalties.py` (state-space constructor may land with the
   lsmr step, since only lsmr consumes it matrix-free).
3. `ridge_lm.py` skeleton: state/info dataclasses, `init`, `update` with the
   `"cholesky"` path only, geodesic, accept/reject, `hyperparams`; direct
   `update`-loop tests.
4. `solve` wiring through the shared loop (callbacks, tolerances with
   TRUE-residual atol, save_steps, multi_start) + `ridge_continuation`.
5. `"qr"` path (cache, damping-row update, CSNE refinement), then `"lsmr"`
   (+ `identity_right_preconditioner`, state-space penalty including the new
   `quasiseparable._cholesky_matvec` scan).
6. Implicit AD (`"cholesky"`, `"normal_cg"`, `"auto"` resolution, failure
   policy) + `test_ridge_ad.py`.
7. Exports, docs page (`docs/ridge_lm.md` with the algorithm exposition and
   references per Sec. 4), README restructure (two headline examples, metric
   downplayed). Full suite + a kernels-repo smoke test (port
   `neoclassical_growth_kernel.py` in a scratch copy, compare solution to
   the metric-LM run).

## Known risks / decisions made

- **Semi-normal equations in the qr path**: mitigated by the fixed
  refinement pass and by `μ`/`λ` bounding the condition number; the
  alternative (carrying Q) doubles Jacobian-sized state. If tests show
  residual stability issues at extreme conditioning, fall back to full
  stacked QR per update (simpler, costlier, still exact) — the switch is
  local to `solve_step`.
- **`loss` now includes the penalty**: any user code that treated
  `info.loss` as `||r||^2` must read `info.resid_loss`; called out in the
  docs and the class docstring.
- **λ in state, not hyper**: parallels `damping`; putting it in
  `LMHyperparams` would break the shared `_cast_hyper` contract for gram_lm.
- **AD is GN-approximate at moderate λ**: accepted and documented (decision:
  Gauss–Newton implicit rule; exact in the interpolating limit; both dropped
  terms and the conditioning caveat stated in the docstring).
- **`penalty_factory` deferred to phase 2** with a differentiation-inert
  data restriction (an `x`- or `p`-dependent `L` invalidates the gradient
  and AD formulas as written); constructor arg reserved, raises
  `NotImplementedError` in phase 1.
- **`M0` materialization fallback is runtime work when penalty data is
  traced**, not a one-time trace-time cost — the cholesky path's
  `add_scaled`-missing fallback must say so in its error/warning text, and
  rejected-step caching saves residual/Jacobian evaluation but NOT the
  per-update GEMM, penalty assembly, or factorization (same trade gram_lm
  makes).
- **ridge = 0 unsupported by contract** (user decision): constructor,
  callback contract, and continuation floors all enforce strict positivity;
  no zero-ridge pseudoinverse tangent is ever needed.

## 8. References

Short tags used throughout this plan; the docs page and docstrings cite the
same works.

- [Bak92] A. B. Bakushinskii, "The problem of the convergence of the
  iteratively regularized Gauss–Newton method," Comput. Math. Math. Phys.
  32(9), 1992, 1353–1359.
- [Bjorck87] Å. Björck, "Stability analysis of the method of seminormal
  equations for linear least squares problems," Linear Algebra Appl. 88–89,
  1987, 31–48.
- [Bjorck96] Å. Björck, *Numerical Methods for Least Squares Problems*,
  SIAM, 1996.
- [BNS97] B. Blaschke (Kaltenbacher), A. Neubauer, O. Scherzer, "On
  convergence rates for the iteratively regularized Gauss–Newton method,"
  IMA J. Numer. Anal. 17(3), 1997, 421–436.
- [CHOS21] Y. Chen, B. Hosseini, H. Owhadi, A. M. Stuart, "Solving and
  learning nonlinear PDEs with Gaussian processes," J. Comput. Phys. 447,
  2021, 110668.
- [CKB12] S. L. Campbell, P. Kunkel, K. Bobinyec, "A minimal norm corrected
  underdetermined Gauß–Newton procedure," Appl. Numer. Math. 62(5), 2012,
  592–605.
- [EHN96] H. W. Engl, M. Hanke, A. Neubauer, *Regularization of Inverse
  Problems*, Kluwer, 1996.
- [EKN89] H. W. Engl, K. Kunisch, A. Neubauer, "Convergence rates for
  Tikhonov regularisation of non-linear ill-posed problems," Inverse
  Problems 5(4), 1989, 523–540.
- [Elden82] L. Eldén, "A weighted pseudoinverse, generalized singular
  values, and constrained least squares problems," BIT 22, 1982, 487–502.
- [FS11] D. C.-L. Fong, M. A. Saunders, "LSMR: An iterative algorithm for
  sparse least-squares problems," SIAM J. Sci. Comput. 33(5), 2011,
  2950–2971.
- [IS26] A. F. Izmailov, M. V. Solodov, "Local convergence of the
  Gauss–Newton methods for constrained nonlinear equations," Comput. Optim.
  Appl., 2026 (doi:10.1007/s10589-026-00801-4).
- [KNS08] B. Kaltenbacher, A. Neubauer, O. Scherzer, *Iterative
  Regularization Methods for Nonlinear Ill-Posed Problems*, de Gruyter, 2008.
- [Marq63] D. W. Marquardt, "An algorithm for least-squares estimation of
  nonlinear parameters," J. SIAM 11(2), 1963, 431–441.
- [More78] J. J. Moré, "The Levenberg–Marquardt algorithm: implementation
  and theory," in *Numerical Analysis* (Dundee 1977), Lecture Notes in
  Math. 630, Springer, 1978, 105–116. (The MINPACK damping-row QR update.)
- [NW06] J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed.,
  Springer, 2006, Ch. 10.
- [PR22] F. Pes, G. Rodriguez, "A doubly relaxed minimal-norm Gauss–Newton
  method for underdetermined nonlinear least-squares problems," Appl. Numer.
  Math. 171, 2022, 233–248.
- [TS12] M. K. Transtrum, J. P. Sethna, "Improvements to the
  Levenberg–Marquardt algorithm for nonlinear least-squares minimization,"
  arXiv:1201.5885, 2012.
