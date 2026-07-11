# Square Systems (Root Finding)

`SquareLevenbergMarquardt` is a solve-only damped-Newton (Levenberg-Marquardt)
root solver for **square nonsingular** systems

$$
r(x, \text{args}, p) = 0,
\qquad \dim r = \dim x ,
$$

built for hot inner loops — the motivating case is the algebraic stage solve
of semi-explicit index-1 DAEs, where a small square root must be found at
every Runge–Kutta stage, warm-started from the previous stage, and
differentiated implicitly with respect to everything that defines the
equation.

It deliberately does not reuse the underdetermined solver's machinery: at
\(m = n\) the residual-space Gram \(J P J^\top\) squares the Jacobian's
condition number for no benefit, and the general solver's metric, callback,
and geodesic plumbing costs real time per stage. The square solver is a lean
jitted `lax.while_loop` around a direct dense factorization.

## API

```python
from nlls_gram import SquareLevenbergMarquardt

solver = SquareLevenbergMarquardt(
    residual_fn,           # (x) | (x, args) | (x, args, p)
    init_damping=1e-3,
    damping_decrease=0.5,
    damping_increase=4.0,
    has_aux=False,
)

result = solver.solve(
    x0,              # root-finding guess only — never an AD target
    args,            # fixed data — never an AD target
    p=p,             # ALL differentiated equation inputs
    max_steps=8,
    atol=None,       # None -> 1e-6 (float32) / 1e-10 (float64)
)
z = result.x         # result.residual_norm, result.steps, result.status
```

There is no `init`/`update` pair — `solve` is the entire interface. The
residual convention, `x` pytree flattening, and `LMStatus` codes match the
underdetermined solver: `x` may be any pytree, and the residual must return a
single array (or `(array, aux)` with `has_aux=True`, where `aux` is a pytree
whose leaves are JAX numeric types — it is returned through the jitted loop)
whose flattened size equals the flattened size of `x`, in the same dtype.
The result is a lean
`SquareSolveResult` with `x`, `residual_norm` (the residual 2-norm at the
returned `x`), `steps`, `status`, `aux` (with `has_aux=True`), and the
pass-through `args` and `p` the solve was called with — the same accessors as
the underdetermined solver's `LMSolveResult`, so downstream code can read
`result.args`/`result.p` uniformly. The loop controls (`max_steps`, `atol`,
`gtol`, `xtol`) are concrete Python scalars, not traceable arguments.

For a DAE stage `0 = g(y, z, t, args, params)` solved for `z`:

```python
def residual(z, args, p):
    y, t, params = p
    return g(y, z, t, args, params)

result = solver.solve(z_guess, args, p=(y, t, params), max_steps=8, atol=root_atol)
z = result.x
```

## The Step: Augmented QR, Never the Gram

Each iteration solves the LM subproblem
\(\min_s \|r + J s\|_2^2 + \lambda \|s\|_2^2\) through one reduced QR of the
augmented matrix

$$
\begin{bmatrix} J \\ \sqrt{\lambda}\, I \end{bmatrix} s
\;\approx\;
\begin{bmatrix} -r \\ 0 \end{bmatrix},
$$

whose normal equations are exactly \((J^\top J + \lambda I)s = -J^\top r\) —
solved without ever forming \(J^\top J\) or \(J J^\top\), so the step does
not square the Jacobian's condition number. For finite \(J\) and
\(\lambda > 0\) the augmented matrix has full column rank (exact arithmetic),
so the factorization does not fail: a singular Jacobian still yields a
well-defined damped step, which the accept/reject test then vets like any
other. Accept/reject on the sum of squared residuals and the multiplicative
damping update follow the main solver.

The Jacobian is a dense forward-mode `jax.jacfwd` and is recomputed only
after an **accepted** step (`lax.cond`); rejected steps reuse it. A warm
start that already meets `atol` exits before the loop without computing a
Jacobian at all — the DAE stage pattern (each stage starts near the next
root) pays one residual evaluation for an already-converged guess (plus one
more at the returned `x` when `has_aux=True`).

The linear algebra is deliberately dense and direct in this first version;
support for other linear solvers may be added later behind a constructor
knob.

## Status Semantics

- `LMStatus.CONVERGED` with the default `gtol=xtol=0` means exactly the
  residual criterion \(\|r\|_2 < \text{atol}\) — a DAE caller can trust the
  status as "found a root," and `result.residual_norm` is there to enforce a
  caller-side criterion directly.
- `gtol` (on \(\|J^\top r\|\)) and `xtol` (on an accepted step norm) are
  opt-in: they can report `CONVERGED` at a stagnation point that is **not** a
  root (e.g. the loss-stationary point of \(x^2 + 1\)), which is precisely
  why they default to off.
- `NONFINITE` reports a nonfinite residual at `x0`; a nonfinite *candidate*
  step is simply rejected (acceptance requires a finite loss). Exhausted
  budgets report `MAX_STEPS`. There are no runtime host exceptions under
  jit; the square-shape and dtype checks raise at trace time.

## Implicit Differentiation

`solve(...).x` carries a custom implicit JVP with respect to `p` (the same
`jax.custom_jvp` organization as the [underdetermined
solver](implicit_ad.md)): differentiate the defining equation at the returned
root rather than the iterations. For \(r(x^*, \text{args}, p) = 0\),

$$
J_x \dot x = -J_p \dot p
$$

is solved **directly** with the square dense \(J_x\) (LU) at the solution —
never through \(J_x J_x^\top\). VJPs come from transposing the JVP rule, so
forward, reverse, forward-over-reverse, and reverse-over-forward all compose
(`jax.hessian`, `jax.vmap` over differentiated solves, etc.). For the DAE
stage above this gives, with no extra rules in the caller,

$$
\dot z = -g_z^{-1}\left(g_y \dot y + g_t \dot t + g_\theta \dot\theta\right).
$$

Contract notes:

- `x0` and `args` receive **zero tangents**: the guess and fixed data are not
  AD targets. Anything that must be differentiated goes in `p`.
- Implicit derivatives are meaningful only when the returned point is a
  converged, **nonsingular** root — check `status` and `residual_norm`
  before trusting them. A singular \(J_x\) produces a non-finite tangent
  (loud), consistent with the library's conventions.
- `result.residual_norm` gets a zero tangent by contract (at a root it is
  ~0 and not meaningfully differentiable); with `has_aux=True`, `result.aux`
  is differentiated through both its direct `p` dependence and the root
  \(x^*(p)\).

## Performance

Construct the residual once at setup scope — solvers compare by
configuration, so a fresh equal-settings instance around the same residual
function reuses the compiled loop, but a residual closure rebuilt per call is
a new configuration and retraces, exactly as with the underdetermined
solver. Measured on CPU (float32, 32
warm-started solves inside one jitted `lax.scan`, the DAE stage pattern;
`benchmarks/test_square_lm_benchmark.py`), the full adaptive LM solve costs
1.0–1.6x a fixed-iteration direct-Newton `fori_loop` baseline at
\(n \in \{1, 4, 8\}\) — within the issue #14 gate of ~2x, while adding
damping robustness, early exit on `atol`, and the implicit-AD boundary.

## API Reference

::: nlls_gram.SquareLevenbergMarquardt

::: nlls_gram.SquareSolveResult
