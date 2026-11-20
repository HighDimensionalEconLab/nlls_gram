# Project: nlls_gram

Levenberg-Marquardt and Gauss-Newton nonlinear least-squares solvers for JAX pytrees.

## Build & Run
- Sync dependencies with `uv sync --group dev --group docs --group benchmark`. CI runs `uv sync --locked`, so update `uv.lock` whenever `pyproject.toml` changes.
- Run checks with `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q`, and `uv run mkdocs build --strict`.

## Versioning & Releases
- Always default to patch version bumps (`x.y.Z`). Never bump minor (`x.Y.0`) or major (`X.0.0`) versions without explicit confirmation from the user, even for API or default changes, while this package primarily serves internal research repos.
- PyPI publishing runs when a GitHub Release is published (`gh release create vX.Y.Z`), not on git tag push alone.
