# Colnade Development Rules

## Before Every Commit

You MUST run the following checks before committing and ensure they all pass:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
uv run ty check tests/typing/ --error-on-warning
```

If any check fails, fix the issue before committing. Do not commit with known failures.
