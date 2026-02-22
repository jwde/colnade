# Colnade Development Rules

## Before Every Commit

You MUST run **all** of the following checks before committing and ensure they all pass:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
uv run ty check tests/typing/ --error-on-warning
uv run python scripts/check_api_docs.py
uv run mkdocs build --strict
```

If any check fails, fix the issue before committing. Do not commit with known failures.

**Important:** Run `ruff check .` and `ruff format --check .` on the entire repo (`.`), not just specific directories. CI runs these against the whole repo.

## Updating Documentation (MANDATORY)

**Any change to a public-facing API MUST include corresponding documentation updates in the same commit.** This is not optional — CI will fail if documentation is missing or inconsistent.

When you add, remove, or rename any public API symbol (anything in `__all__`):

1. **API reference** (auto-generated) — Add/update/remove the `::: module.Symbol` directive in the appropriate `docs/api/*.md` file. Run `uv run python scripts/check_api_docs.py` to verify coverage. **This check runs in CI and will block merging if it fails.**
2. **User guide** (manual) — Update the relevant page in `docs/user-guide/` to document new features or reflect removed ones.
3. **Tutorials** (manual) — Update any affected tutorials in `docs/tutorials/` if they reference changed APIs.
4. **README.md** — Update if the change affects the quick start, feature list, or examples shown to new users.
5. **mkdocs build** — Run `uv run mkdocs build --strict` to verify docs build without warnings.
