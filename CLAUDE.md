# Colnade Development Rules

## Before Every Commit

You MUST run the following checks before committing and ensure they all pass:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
uv run ty check tests/typing/ --error-on-warning
uv run python scripts/check_api_docs.py
```

If any check fails, fix the issue before committing. Do not commit with known failures.

## Updating Documentation

When you add, remove, or rename any public API symbol (anything in `__all__`):

1. **API reference** (auto-generated) — Add/update/remove the `::: module.Symbol` directive in the appropriate `docs/api/*.md` file. Run `uv run python scripts/check_api_docs.py` to verify coverage.
2. **User guide** (manual) — Update the relevant page in `docs/user-guide/` to document new features or reflect removed ones.
3. **Tutorials** (manual) — Update any affected tutorials in `docs/tutorials/` if they reference changed APIs.
