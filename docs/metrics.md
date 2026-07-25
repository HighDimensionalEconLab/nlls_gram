# Metrics and preconditioners

The two hook types both solvers take. They look similar and mean opposite
things:

- a **`Metric`** defines the subproblem, so it must be **exact**;
- a **`Preconditioner`** only changes the CG iteration path, so it may
  approximate freely.

Instances of both are **JAX pytrees**: array fields are traced leaves, the
type plus its static fields are structure. They ride inside the solver state
(`lm_state.metric`, `lm_state.preconditioner`), so a `solve` callback adapts
one by **calling the constructor again** with fresh arrays — same type, same
leaf shapes and dtypes, pure traced ops, no recompilation. See
[Callbacks](callbacks.md) for the adaptation rules.

## `Metric`

A positive-definite \(W\) given through callbacks for an invertible factor
\(F\) with \(W = F^\top F\). The solver runs in the whitened variable and
never materializes either.

```python
class Metric:
    size: int             # static: the metric block, the leading coordinates of x
    free_scale = 1.0      # traced leaf: damping weight on everything past it

    def factor_apply(v, ctx): ...           # F v
    def factor_solve(v, ctx): ...           # F^-1 v
    def factor_solve_transpose(v, ctx): ... # F^-T v
    def norm(v, ctx): ...                   # ||F v||, defaulted
```

Ops read `self` and act on metric-block vectors, or matrices whose *leading*
axis is `size` (columns are batched). `ctx` is a `SolverContext` carrying the
flat iterate and the live `LMState`, so an exotic metric can key off either.
Shipped implementations:

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
| `BlockEigenPreconditioner(families, permutation)` | a block-diagonal eigenbasis, analytic in both ridge and damping |
| `NystromPreconditioner(matvec, n, rank, key)` | a randomized rank-\(k\) sketch (Frangella-Tropp-Udell) |
| `ShermanMorrisonPreconditioner(solve, u, weight)` | \(A + wuu^\top\) from a solve with \(A\) |
| `WoodburyPreconditioner(solve, U, weights)` | the rank-\(k\) generalization |
| `PaddedPreconditioner(base, n_real)` | a base extended over exactly-zero padded rows |

Expensive derived state (`BlockEigen`'s eigendecompositions, `Nystrom`'s
sketch) is computed once in the constructor and stored as leaves; refreshing
mid-solve means constructing a new instance in the callback, usually behind
`jax.lax.cond` so the cost is paid only when the refresh fires. Staleness is
always safe — it moves the CG iteration path, never the converged step.

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

## Custom types

A custom metric or preconditioner is a small frozen dataclass registered with
`register_pytree_dataclass` — the solvers reject unregistered instances:

```python
from dataclasses import dataclass

@dataclass(frozen=True, eq=False)
class JacobiPreconditioner(Preconditioner):
    diagonal: jax.Array

    def apply(self, v, damping, ctx):
        return v / (self.diagonal + damping)

register_pytree_dataclass(JacobiPreconditioner, data_fields=("diagonal",))
```

`data_fields` become traced leaves; `meta_fields` (compile-time structure —
sizes, block counts) must be hashable. The constructor may validate and
compute derived leaves eagerly: reconstruction inside jit or a loop carry
restores the stored fields verbatim without re-running it. Constructors must
be traceable when the instance is rebuilt inside a jitted callback. Metrics
also support `dataclasses.replace` (their constructors are cheap);
preconditioners are rebuilt by constructor only.

## Compilation

Instances key compilation by pytree **structure** — the type plus its static
fields — while their arrays are threaded through the carried state. Two
solvers around equal-config instances with fresh arrays share one compiled
loop, so constructing a metric per solve call is free; a changed static
field (a different `repeats`, a different family layout) keys a new program.
`tests/test_compilation.py` pins this.
