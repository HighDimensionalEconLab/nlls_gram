# Metrics and preconditioners

The two hook types both solvers take. They look similar and mean opposite
things:

- a **`Metric`** defines the subproblem, so it must be **exact**;
- a **`Preconditioner`** only changes the CG iteration path, so it may
  approximate freely.

## `Metric`

A positive-definite \(W\) given through callbacks for an invertible factor
\(F\) with \(W = F^\top F\). The solver runs in the whitened variable and
never materializes either.

```python
class Metric:
    size: int             # the metric block: the leading coordinates of x
    free_scale = 1.0      # damping weight on everything past it

    def factor_apply(v, ctx): ...           # F v
    def factor_solve(v, ctx): ...           # F^-1 v
    def factor_solve_transpose(v, ctx): ... # F^-T v
    def norm(v, ctx): ...                   # ||F v||, defaulted
```

Ops act on metric-block vectors, or matrices whose *leading* axis is `size`
(columns are batched). Shipped implementations:

| | \(W\) |
|---|---|
| `IdentityMetric(size)` | \(I\) |
| `CholeskyMetric(L)` | \(LL^\top\) |
| `DiagonalMetric(weights)` | \(\operatorname{diag}(w)\) |
| `RepeatedFactorMetric(F, repeats=r)` | \(\operatorname{blockdiag}(F^\top F, \ldots)\) |

`RepeatedFactorMetric` shares one triangular solve across every repeated block
by packing them into the columns of a single right-hand side, so no repeated
factor or full block diagonal is ever formed.

## `Preconditioner`

An SPD approximation of the damped inverse, for the Krylov configs.

```python
class Preconditioner:
    def apply(v, damping, ctx): ...   # ~ (operator + damping I)^-1 v
```

Which space `v` lives in is named by the config: `CG` is parameter space,
`GramCG` residual space. `IdentityPreconditioner()` is the explicit opt-out —
running Krylov methods without a preconditioning decision should be a visible
choice, not a default.

| | approximates |
|---|---|
| `IdentityPreconditioner()` | \(I\) |
| `BlockEigenPreconditioner(blocks_fn, permutation)` | a block-diagonal eigenbasis, analytic in both ridge and damping |
| `NystromPreconditioner(matvec, n, rank, key)` | a randomized rank-\(k\) sketch (Frangella-Tropp-Udell) |
| `ShermanMorrisonPreconditioner(solve, u, weight)` | \(A + wuu^\top\) from a solve with \(A\) |
| `WoodburyPreconditioner(solve, U, weights)` | the rank-\(k\) generalization |
| `PaddedPreconditioner(base, n_real)` | a base extended over exactly-zero padded rows |

**Range preservation.** On rank-deficient problems the minimum-norm selection
rests on the CG iterates staying in \(\operatorname{range}(B^\top)\).
Unpreconditioned CG from zero does, since the right-hand side starts there and
the operator maps the subspace to itself. A preconditioner \(C\) enters the
Krylov space through its images, so unless
\(C(\operatorname{range}(B^\top)) \subseteq \operatorname{range}(B^\top)\) the
iterates leak into the null space and the step silently stops being the
minimum-norm one — the CG residual still converges, so nothing fails loudly.
Safe: the identity, polynomials in the operator, an exact \((B^\top B + \tau
I)^{-1}\) at fixed \(\tau > 0\). On full-column-rank problems the condition is
vacuous.

## Iterate-dependent state

Both types take the same optional pair when their numbers must track the
iterate:

```python
def prepare(self, theta, ctx): ...   # -> traced pytree, or None (the default)
def rebuild(self, ctx): ...          # -> traced bool, True by default
```

The output rides on the solver state and comes back as `ctx.metric_state` /
`ctx.preconditioner_state`. It is rebuilt on accepted steps and reused across
rejected ones (where `x` did not move), runs inside the jitted loop as traced
ops, and is **frozen at the solution** under implicit AD. Its pytree structure
must not change between rebuilds. Override `rebuild` to decline a refresh —
for a preconditioner that is always safe and often much cheaper.

Setup that does *not* depend on the iterate belongs in `__init__`, where it is
paid once.

## Compilation

The solver is a jit **static** argument, so its hooks enter the compile cache
key. Types holding arrays (every metric above, most preconditioners) hash by
identity: **build them once at setup scope and reuse them**, or every
construction keys a fresh compilation of the whole solve loop. Stateless
value-equal types (`IdentityPreconditioner`) are free to construct inline.
`tests/test_compilation.py` pins this.
