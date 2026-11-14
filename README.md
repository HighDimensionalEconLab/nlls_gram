# nlls_gram

[![CI](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml/badge.svg)](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml)
[![Docs](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/docs.yml/badge.svg)](https://highdimensionaleconlab.github.io/nlls_gram/)
[![PyPI](https://img.shields.io/pypi/v/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![Python versions](https://img.shields.io/pypi/pyversions/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![License: MIT](https://img.shields.io/github/license/HighDimensionalEconLab/nlls_gram)](https://github.com/HighDimensionalEconLab/nlls_gram/blob/main/LICENSE)

Euclidean Levenberg-Marquardt and Gauss-Newton nonlinear least squares for JAX pytrees, with generalized Tikhonov regularization, inequality and complementarity constraints, and implicit differentiation.

## Setup

```bash
uv add nlls-gram
```

For CUDA GPUs:

```bash
uv add nlls-gram "jax[cuda13]"
```

For local development:

```bash
uv sync --group dev --group docs --group benchmark
```

## Run

```python
import jax
import jax.numpy as jnp
from nlls_gram import AnnealRidge, LevenbergMarquardt, RepeatedFactorRidge


def residual(x, args, p):
    return args["design"] @ x - p["target"]


solver = LevenbergMarquardt(residual)
result = solver.solve(
    jnp.zeros(8), {"design": design}, p={"target": y}, max_steps=200, atol=1e-8
)

grad_p = jax.grad(
    lambda p: jnp.sum(
        solver.solve(jnp.zeros(8), {"design": design}, p=p, max_steps=200, atol=1e-8).x
        ** 2
    )
)({"target": y})
```

Generalized Tikhonov regularization with `RepeatedFactorRidge`:

```python
ridge_metric = RepeatedFactorRidge(jnp.linalg.cholesky(K, upper=True), repeats=2)
solver = LevenbergMarquardt(collocation_residual, ridge=1e-4, ridge_metric=ridge_metric)
anneal = AnnealRidge(ridge_floor=1e-10)
result = solver.solve(x0, callback=anneal, user_state=anneal.init_state(), gtol=1e-8)
```

Run tests, benchmarks, and docs:

```bash
uv run pytest -q
uv run pytest benchmarks/test_benchmarks.py --benchmark-only
uv run mkdocs build --strict
```

## Documentation

<https://highdimensionaleconlab.github.io/nlls_gram/>

## License

MIT.
