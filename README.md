# nlls_gram

[![CI](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml/badge.svg)](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml)
[![Docs](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/docs.yml/badge.svg)](https://highdimensionaleconlab.github.io/nlls_gram/)
[![PyPI](https://img.shields.io/pypi/v/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![Python versions](https://img.shields.io/pypi/pyversions/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![License: MIT](https://img.shields.io/github/license/HighDimensionalEconLab/nlls_gram)](https://github.com/HighDimensionalEconLab/nlls_gram/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Levenberg-Marquardt nonlinear least-squares for JAX pytrees, aimed at
solving systems of equations (i.e., interpolation). The core use case is
underdetermined systems — more parameters than residuals — where something
must select *which* interpolating solution is returned. The package ships
two solvers with one shared `init`/`update`/`solve` protocol:

- **`RidgeLevenbergMarquardt`** puts the selection in the **objective**: it
  minimizes \(\|r(x)\|^2 + \lambda\,\|Lx\|^2\) for a user penalty factor
  \(L\) with the ridge weight \(\lambda\) annealed toward zero, so the
  limit is the minimum-seminorm (e.g. minimum-RKHS-norm) interpolant by
  classical nonlinear Tikhonov regularization ([Engl–Hanke–Neubauer 1996;
  Kaltenbacher–Neubauer–Scherzer 2008](https://highdimensionaleconlab.github.io/nlls_gram/ridge_lm/)).
  Every inner problem is a well-posed NLLS — plain Gauss-Newton alone would
  converge to *some* interpolant without selecting the minimal-norm one
  ([Campbell–Kunkel–Bobinyec 2012](https://highdimensionaleconlab.github.io/nlls_gram/ridge_lm/)).
- **`LevenbergMarquardt`** is standard damped LM for general nonlinear least
  squares — square, tall, or underdetermined — with dense, QR, CG, and
  matrix-free LSMR linear solvers and a swappable implicit-AD rule.

Both take a residual over `(x)`, `(x, args)`, or `(x, args, p)` (always in
that order), flatten any pytree `x` with `jax.flatten_util.ravel_pytree`,
expose per-step `update(...)` and an internally jitted `solve(...)` loop
with callbacks and multi-start, and differentiate `solve(...).x` with
respect to `p` through a custom implicit rule.

## Install

```bash
uv add nlls-gram
```

For GPU use, install the JAX accelerator build that matches your hardware, for
example:

```bash
uv add nlls-gram "jax[cuda13]"
```

## Ridge Example: Minimum-RKHS-Norm Interpolation

For kernel coefficient problems
\(f_\alpha(x)=\sum_j \alpha_j K(x, x_j)\), the squared RKHS norm is
\(\alpha^\top K \alpha\), so the penalty is `repeated_dense_penalty`
(`repeats` coefficient blocks sharing one Gram matrix `K`, plus unpenalized
structural scalars — no epsilon shift anywhere):

```python
import jax.numpy as jnp

from nlls_gram import (
    RidgeLevenbergMarquardt,
    repeated_dense_penalty,
    ridge_continuation,
)

penalty = repeated_dense_penalty(K, repeats=3, zero_pad_size=d)
solver = RidgeLevenbergMarquardt(
    residual_fn,                 # (x) | (x, args) | (x, args, p)
    penalty=penalty,
    ridge=1e-8,                  # fixed small ridge; None = dtype default
    linear_solve_dtype=jnp.float64,
)
result = solver.solve(theta_0, max_steps=400, gtol=1e-8, atol=1e-8)

# optional homotopy (ridge annealed 1e-4 -> 1e-8 on stationarity):
cb, us0 = ridge_continuation(ridge_floor=1e-8, decrease=0.1)
result = solver.solve(theta_0, max_steps=400, gtol=1e-8, atol=1e-8,
                      callback=cb, user_state=us0)
```

The solver is stock Euclidean LM on the augmented residual
\([r;\sqrt{\lambda}Lx]\): the trust-region damping and the selection
weight are fully decoupled, the assembled normal matrix is cached across
rejected steps, and a damping-row QR path stays accurate at tiny ridge.
`info.loss` is the ridge objective; `info.resid_loss` is the equation error.
Stopping is conjunctive: `gtol` means "stationary at this ridge", `atol`
additionally demands the equations solved (it never stops the solve alone).
The [Ridge LM docs](https://highdimensionaleconlab.github.io/nlls_gram/ridge_lm/)
derive the selection theorem, the continuation schedule, and the solver
table.

## General Nonlinear Least Squares

```python
import jax
import jax.numpy as jnp

from nlls_gram import LevenbergMarquardt


def residual_fn(x, args):
    ts, ys = args
    return x["a"] * jnp.exp(x["b"] * ts) - ys


ts = jnp.linspace(0.0, 2.0, 20)
ys = 2.0 * jnp.exp(-1.0 * ts)
x = {"a": 1.0, "b": 0.0}

solver = LevenbergMarquardt(residual_fn, init_damping=1e-2)
lm_state = solver.init(x, (ts, ys))


@jax.jit
def train_step(x, lm_state):
    return solver.update(x, lm_state, (ts, ys))


for _ in range(50):
    x, lm_state, info = train_step(x, lm_state)

print(x["a"], x["b"])  # approximately 2.0, -1.0
```

For a simple full solve loop:

```python
result = solver.solve(x, (ts, ys), max_steps=50, atol=1e-8)
x = result.x
```

`solve` stops on a residual-norm `atol`, gradient-norm `gtol`, or
accepted-step-norm `xtol` (each `0.0` disables), always enforces `max_steps`,
and takes a traceable callback for custom stopping, epoch-style data
resampling, and per-step history recording; the docs have a cookbook.

`solve(...).x` also supports custom implicit JVP/VJP with respect to `p`;
the docs give the metric-minimum-norm formula and a minimal `jax.jvp` /
`jax.vjp` example. The default `ad_solver="auto"` uses a direct solve for
every square system, preserves the forward CG space for nonsquare CG systems,
and uses SVD otherwise. Every method is independently swappable (an `lsmr` forward solve with
`ad_solver="normal_cg"` is fully matrix-free end to end). The metric
matters for underdetermined roots because it selects which tangent is the
minimum-norm solution. The per-step `update(...)` interface does not define
the implicit AD rule. By default, both `CONVERGED` and `MAX_STEPS` results are
usable: fixed-step solves retain their implicit derivative while the status
still reports `MAX_STEPS`. Pass `max_steps_is_success=False` for strict
failure semantics. A failed solve keeps its primal result and diagnostics but
contributes exactly zero through `result.x` and `result.aux`; `result.p`
remains an identity pass-through. Its linear tangent program is evaluated
safely at differentiation-inert copies of the caller's original
`(x0, args, p)`, so those initial inputs must be JVP-safe for the residual,
aux map, and any metric or preconditioner factory used by the selected AD
method. This also keeps mixed successful/failed `vmap` lanes finite.

### Advanced: metric damping

`LevenbergMarquardt` optionally accepts a positive-definite parameter-space
`metric` (or an iterate-aware `MetricFactory`) that redefines the damping
geometry, so the small-damping Gauss-Newton limit selects
minimum-*metric*-norm corrections — the algorithmic-bias alternative to the
ridge objective, with `repeated_shifted_dense_metric` /
`repeated_shifted_state_space_metric` covering the kernel geometry
\(\operatorname{blockdiag}(K,\ldots,K,0_s)+\varepsilon I\). See the
[metrics](https://highdimensionaleconlab.github.io/nlls_gram/metrics/) and
[metric Gauss-Newton](https://highdimensionaleconlab.github.io/nlls_gram/gauss_newton/)
docs pages; for new kernel problems prefer `RidgeLevenbergMarquardt`.

## LevenbergMarquardt Solvers

- `linear_solver="auto"` (the default): resolves at trace time to the
  smaller dense factorization — `gram_cholesky` when `n > m`,
  `normal_cholesky` otherwise. A shape rule, and safely so: the two forms
  compute the same step.
- `linear_solver="gram_cholesky"`: dense `m × m` residual-space Gram solve.
- `linear_solver="normal_cholesky"`: dense `n × n` whitened normal solve;
  its small-damping limit is the minimum-metric-norm least-squares step at
  every shape and rank.
- `linear_solver="qr"`: dense QR solve of the whitened-step problem (requires
  a full-row-rank Jacobian).
- `linear_solver="augmented_qr"`: direct augmented QR in parameter space;
  robust to rank-deficient Jacobians when damping is positive and best suited
  to small systems.
- `linear_solver="gram_cg"`: matrix-free residual-space CG. A
  `dual_preconditioner` is required (e.g. `sherman_morrison_preconditioner`,
  or the randomized `nystrom_preconditioner` for neural-network duals; pass
  `identity_preconditioner()` to run unpreconditioned CG explicitly);
  on a nonsquare system `ad_solver="auto"` keeps `solve(...).x` matrix-free
  under AD and requires `ad_solver_preconditioner` when the differentiated
  solve is traced (an explicit `ad_solver="gram_cg"` validates it eagerly).
  When the dual operator rotates as LM
  drifts `x`, pass `preconditioner_factory=PreconditionerFactory(prepare,
  apply)` instead — a θ-adaptive preconditioner rebuilt from the live iterate
  each step — and `recycle=RecycleConfig(rank=k)` to carry a deflation basis
  across steps, recycling each solve's Krylov subspace into the next.
- `linear_solver="normal_cg"`: matrix-free CG on the whitened normal system,
  iterating in parameter space — the matrix-free form for square-to-tall
  problems. A `normal_preconditioner` is required; on rank-deficient
  problems it must preserve `range(Bᵀ)` or the minimum-norm selection is
  lost (`identity_preconditioner()` always qualifies — the docs give the
  full requirement).
- `linear_solver="lsmr"`: matrix-free LSMR on the whitened augmented system,
  the iterative sibling of `augmented_qr`, using only J/Jᵀ products. It works
  on the whitened Jacobian rather than a squared Gram/normal operator, so it
  stays accurate at small damping where those solves hit their `eps·cond`
  floor. An optional
  `whitened_preconditioner=WhitenedPreconditioner(solve, solve_transpose)`
  right-preconditions the operator to cluster its spectrum; every damped
  posed subproblem stays exactly the identity-damped whitened one, so the
  preconditioner changes the iteration path, never the converged step.

All eight solve the same metric-damped linearized subproblem up to the
accuracy of the chosen linear solver. The dense paths materialize the
Jacobian from its small side (`jacobian_mode="auto"`; `"fwd"`/`"rev"`
force one AD mode), so tall systems never build an `m × m` residual basis.

## Docs and Alternatives

Full docs: https://highdimensionaleconlab.github.io/nlls_gram/

Working with an AI assistant? Point it at
[`docs/tuning_guide.md`](https://highdimensionaleconlab.github.io/nlls_gram/tuning_guide/)
if it doesn't pick it up automatically — solver selection, damping heuristics,
inner-solve scheduling, and failure signatures, written to be read by humans
and agents alike (also indexed via the site's `llms.txt`).

For a broader JAX nonlinear solver library, see
[Optimistix](https://github.com/patrick-kidger/optimistix). `nlls_gram` is more
specialized: it focuses on underdetermined nonlinear least-squares, residual
space Gram solves, and explicit parameter-space metrics.

## License

MIT
