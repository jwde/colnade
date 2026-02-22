# Contributing to Colnade

Thanks for your interest in contributing! This guide covers the development workflow.

## Setup

Colnade uses [uv](https://docs.astral.sh/uv/) for dependency management in a monorepo workspace (core + Polars/Pandas/Dask backends).

```bash
git clone https://github.com/jwde/colnade.git
cd colnade
uv sync
```

This installs all packages in the workspace plus dev dependencies (pytest, ruff, ty, mkdocs).

## Running checks

All of these must pass before submitting a PR:

```bash
uv run ruff check .              # Lint
uv run ruff format --check .     # Format
uv run pytest tests/ -v          # Tests (1087+)
uv run ty check tests/typing/ --error-on-warning  # Static type tests
uv run python scripts/check_api_docs.py            # API docs completeness
uv run mkdocs build --strict     # Docs build
```

To run tests with coverage:

```bash
uv run pytest tests/ --cov=colnade --cov=colnade_polars --cov=colnade_pandas --cov=colnade_dask --cov-report=term-missing
```

## Project structure

```
src/colnade/          Core library (schemas, expressions, DataFrame, validation)
colnade-polars/       Polars backend adapter + I/O
colnade-pandas/       Pandas backend adapter + I/O
colnade-dask/         Dask backend adapter + I/O
tests/
  unit/               Unit tests for core library
  integration/        Integration tests across backends
  typing/             Static type tests checked by ty
examples/             Runnable example scripts
docs/                 MkDocs documentation site
scripts/              Maintenance scripts (API docs check, error showcase generator)
```

## Pull requests

- Create a feature branch from `main`
- Keep PRs focused — one feature or fix per PR
- If your change adds or modifies a public API symbol, update the documentation (see CLAUDE.md for the full checklist)
- Add an entry to `CHANGELOG.md` under `[Unreleased]` for user-visible changes
- CI must pass before merging

## Code style

- Ruff handles linting and formatting (config in `pyproject.toml`)
- Line length: 100 characters
- Target: Python 3.10+
- Use type annotations throughout

## Adding a new backend

Backend adapters implement the protocol in `src/colnade/_protocols.py`. See `colnade-polars/` for the reference implementation. Each backend is a separate package in the workspace with its own `pyproject.toml`.
