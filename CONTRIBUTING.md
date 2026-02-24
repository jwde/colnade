# Contributing to Colnade

Thanks for your interest in contributing!

## How to contribute

### Reporting bugs or requesting features

Open a [GitHub issue](https://github.com/jwde/colnade/issues). New issues from external contributors are labeled `triage` and reviewed by a maintainer. Once confirmed, issues are moved to `approved`.

**Please do not start coding until an issue is `approved`** — this avoids wasted effort on work that may not align with the project direction.

### Small fixes (typos, docs, obvious bugs)

Open a PR directly — no issue needed. Keep it focused.

### New features, API changes, or new backends

1. **Open an issue first** describing what you want to do and why
2. **Wait for maintainer response** — we'll discuss the approach and confirm it's `approved`
3. **Then open a PR** referencing the issue

### Finding something to work on

Look for issues labeled [`approved`](https://github.com/jwde/colnade/issues?q=is%3Aissue+is%3Aopen+label%3Aapproved) — these are confirmed and ready for someone to pick up. Issues also labeled [`good first issue`](https://github.com/jwde/colnade/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) are scoped for newcomers.

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
uv run pytest tests/ -v          # Tests
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
- Reference the GitHub issue in the PR description
- Keep PRs focused — one feature or fix per PR
- If your change adds or modifies a public API symbol, update the documentation (see below)
- Add an entry to `CHANGELOG.md` under `[Unreleased]` for user-visible changes
- CI must pass before merging

### Documentation requirements

Any change to a public API must include documentation updates in the same PR:

1. **API reference** — add/update the `::: module.Symbol` directive in `docs/api/*.md`
2. **User guide** — update the relevant page in `docs/user-guide/`
3. **README.md** — update if the change affects the quick start or feature list

Run `uv run python scripts/check_api_docs.py` to verify API docs are complete.

## Code style

- Ruff handles linting and formatting (config in `pyproject.toml`)
- Line length: 100 characters
- Target: Python 3.10+
- Use type annotations throughout

## Adding a new backend

Backend adapters implement the protocol in `src/colnade/_protocols.py`. See `colnade-polars/` for the reference implementation. Each backend is a separate package in the workspace with its own `pyproject.toml`. Open an issue to discuss before starting.
