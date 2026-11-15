# nlls_gram

[![CI](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml/badge.svg)](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/ci.yml)
[![Docs](https://github.com/HighDimensionalEconLab/nlls_gram/actions/workflows/docs.yml/badge.svg)](https://highdimensionaleconlab.github.io/nlls_gram/)
[![PyPI](https://img.shields.io/pypi/v/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![Python versions](https://img.shields.io/pypi/pyversions/nlls-gram.svg)](https://pypi.org/project/nlls-gram/)
[![License: MIT](https://img.shields.io/github/license/HighDimensionalEconLab/nlls_gram)](https://github.com/HighDimensionalEconLab/nlls_gram/blob/main/LICENSE)

Levenberg-Marquardt and Gauss-Newton nonlinear least squares for JAX pytrees, with generalized Tikhonov regularization, inequality and complementarity constraints, and implicit differentiation.

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
import jax.numpy as jnp
from nlls_gram import LevenbergMarquardt

design = jnp.arange(24.0).reshape(4, 6)
y = jnp.ones(4)

solver = LevenbergMarquardt(lambda x, args, p: args["A"] @ x - p["y"])
result = solver.solve(jnp.zeros(6), {"A": design}, p={"y": y}, atol=1e-6)
```

Run tests, benchmarks, and docs:

```bash
uv run pytest -q
uv run pytest benchmarks/test_benchmarks.py --benchmark-only
uv run mkdocs build --strict
```

Documentation: <https://highdimensionaleconlab.github.io/nlls_gram/>
