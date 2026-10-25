# Row-wise residuals

A residual built from a simulated path has the shape

$$
r(\theta) = \big[\, g(x_s(\theta), \theta) \,\big]_{s=1}^{m},
$$

where \(x_s(\theta)\) is state \(s\) of a path driven by the parameters (a
`lax.scan`, so every row depends on \(\theta\) through the path) and \(g\)
evaluates one equation at one state, typically a network at that state and at
a batch of successor states. The natural implementation is `jax.vmap(g)` over
the stacked states, and that `vmap` is where the cost of the Jacobian hides.

## Where the cost comes from

`LevenbergMarquardt` linearizes \(r\) at \(\theta\) once per step and
assembles \(J\) from the small side: \(n\) unit tangents through the
linearization when \(n \le m\), otherwise the \(m\) unit cotangents through
its transpose, in one `vmap` either way. For the fat case that is

```python
resid, jvp = jax.linearize(residual, theta)
transpose = jax.linear_transpose(jvp, theta)
J = jax.vmap(lambda c: transpose(c)[0])(jnp.eye(m))
```

The transpose of `vmap(g)`'s linearization at the cotangent \(e_s\) applies
the pullback of \(g\) at every row \(s'\) with cotangent \(e_s[s']\), which
is zero for every \(s' \ne s\). Over the \(m\) cotangents that is \(m^2\)
row pullbacks, \(m\) of them carrying a signal, and XLA keeps the batched
intermediates of all of them live: with \(Q\) successor states of dimension
\(N\) and a width-\(w\) network, \(m \times m \times Q \times N \times w\)
numbers. The tall case has the same shape with the roles swapped: \(n\)
unit tangents, each pushing the tangent of \(g\) through all \(m\) rows,
\(mn\) row passes in total. The path part of the derivative (the `scan`) is
one sweep per unit tangent or cotangent in both cases.

The solver cannot remove this: it sees \(r\) as a black box from \(\theta\)
to \(\mathbb{R}^m\), and the row structure lives inside the residual.
Checkpointing or `lax.map` over the rows trade the memory for recomputation
or serialization and keep the \(m^2\) passes.

## The rule

`vmap_rows` is `jax.vmap` with a `custom_jvp`. Writing \(z_s\) for the
inputs of row \(s\) that carry a tangent (its slice of the mapped arguments
and the unmapped ones, typically \(x_s\) and \(\theta\)), the value is
`vmap(g)` and the tangent rule is

$$
\dot r_s = G_s\, \dot z_s, \qquad G_s = \frac{\partial g}{\partial z}(z_s),
$$

with \(G_s\) computed at the linearization point by `jax.jacrev` vmapped
over the rows: one reverse pass per row for a scalar \(g\), one per output
entry otherwise. Arguments without a tangent (data closed over by the
residual, integer and boolean leaves) are left out of \(z_s\) and get no
stored Jacobian. `jax.linearize` keeps \(G_s\) as a residual of the
linearization, so what the assembly maps over is the contraction, not
\(g\): a tangent column costs \(G_s \dot z_s\) per row, and the pullback of a
cotangent \(c\) is \(G_{s,x}^\top c_s\) for the mapped inputs and
\(\sum_s G_{s,\theta}^\top c_s\) for the shared ones, a batched matrix
product either way. For a scalar \(g\):

| | `jax.vmap` | `vmap_rows` |
|---|---|---|
| passes of \(g\) per Jacobian, \(m < n\) | \(m^2\) pullbacks | \(m\) pullbacks |
| passes of \(g\) per Jacobian, \(n \le m\) | \(mn\) tangents | \(m\) pullbacks |
| stored for the assembly | the batched intermediates of \(g\) | \(m \times \dim z\) |

The table counts passes of \(g\). The compiled temporaries also hold the
contractions and the cotangents handed to the path, \(m \times m \times
\dim x_s\) over the unit cotangents, which is the same in both columns.

The rule is exact. \(J\) is the same matrix, Jacobian-vector products are
the same, and plain directional `jvp` calls outside `jax.linearize` -- the
geodesic second directional derivative `jvp(jvp(r))` and the implicit-AD
tangent \(\partial r / \partial p\) -- take the native forward-mode
`jvp(vmap(g))` path directly without building row Jacobians.

## Chunking

When the row function itself has large per-row reverse-mode intermediates,
`vmap_rows(..., chunk_size=k)` evaluates the per-row `jacrev` pass under
`jax.linearize` via `jax.lax.map(..., batch_size=k)` instead of a single
`vmap` over all \(m\) rows, while keeping the forward evaluation as `vmap`.

What `vmap_rows` leaves outside the row block is the path: the unit
cotangents carried backward through the simulation `scan`, an \((m, m, N)\)
array plus the matching activations. `jacobian_chunk=c` on the solver pushes
the basis through the linearization in `lax.map` chunks of \(c\) instead of
one `vmap`, so that array becomes \((c, m, N)\). Measured on an H100 at
\(N = 16384\) with \(m = 256\) rows: one batch holds 16.7 GB of temporaries
(largest array 8.6 GB) in 1.36 s; chunks of 64 hold 11.9 GB (largest 2.1 GB)
in 1.40 s. Smaller chunks save nothing more, since the row block's own
activations, linear in \(m\), are then the floor (unless chunked with
`chunk_size` on `vmap_rows`).

## Using it

```python
import jax.numpy as jnp
from flax import nnx
from nlls_gram import LevenbergMarquardt, vmap_rows


@vmap_rows(in_axes=(0, 0, None, None, None, None))
def euler_residual(X, u, next_shocks, weights, policy, p):
    X_next = u + (1 - p["delta"]) * X[None] + next_shocks
    u_next = jax.vmap(policy)(X_next)
    return p["gamma"] * u - p["beta"] * (weights @ (price(X_next, p) + u_next))


def residual(theta, args, p):
    policy = nnx.merge(args.graphdef, theta)
    X, u = simulate(args.X_0, args.shocks, policy, p)
    r = euler_residual(X, u, args.next_shocks, args.weights, policy, p)
    return r / jnp.sqrt(r.size)


solver = LevenbergMarquardt(residual)
```

The decorator replaces `jax.vmap` on the row function and nothing else
changes. Its requirements on `f`:

- Rows are independent: output row \(s\) depends only on row \(s\) of the
  mapped arguments.
- Arguments are positional, and everything `f` differentiates arrives as
  one. An `nnx` module, a parameter dict, and integer or boolean arrays are
  all fine; a closed-over tracer raises.
- The output is a scalar or a small pytree of real floating leaves, in the
  dtype of the differentiated inputs (the stored Jacobians take the input
  dtype). A row that returns \(k\) entries stores \(k\) Jacobians. Auxiliary
  outputs for `has_aux=True` belong in the residual around the block, not in
  `f`.

It composes with `jit`, `scan`, and an outer `vmap`.

## Measured

`benchmarks/test_vmap_rows_benchmark.py` assembles the Jacobian of a
simulated-path residual (\(m = 64\) states of \(N\) firms, \(Q = 3\)
successors, a width-16 two-layer policy, \(n = 321\)) under both
decorators. Temporary memory of the compiled program and median time:

| | `jax.vmap` | `vmap_rows` |
|---|---|---|
| CPU float64, \(N = 128\) | 814 MB, 38 ms | 22 MB, 15 ms |
| CPU float64, \(N = 1024\) | 6.5 GB, 360 ms | 171 MB, 54 ms |
| RTX 3090 float32, \(N = 1024\) | 1.6 GB, 17 ms | 60 MB, 6 ms |
