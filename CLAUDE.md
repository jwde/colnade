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

## Architectural Decisions

### Column[DType] Annotation Pattern

Schema column annotations use `Column[DType]` instead of bare dtype annotations:

```python
# Correct — type checker sees Column methods
class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt8 | None]

# WRONG — type checker sees bare dtype, no Column methods visible
class Users(Schema):
    age: UInt8
```

**Rationale:** All Python type checkers (ty, mypy, pyright) read static annotations, not runtime metaclass replacements. With bare `age: UInt8`, the type checker thinks `Users.age` is `UInt8` — a sentinel class with no `.sum()`, `.mean()`, etc. The `Column[DType]` pattern (inspired by SQLAlchemy 2.0's `Mapped[T]`) makes the annotation _be_ the descriptor type.

### Self Narrowing Limitation

The spec envisions dtype-conditional method availability using `self` type narrowing (e.g. `.field()` only on Struct columns, `.list` only on List columns). ty does not yet support `self` narrowing on non-Protocol generic classes, so these methods are available on all `Column` instances at the type level.

**What returns `Any` (improvable with self narrowing):**
- `.list` property → `ListAccessor[Any]` (should be `ListAccessor[T]`)
- `.get()`, `.sum()`, `.mean()`, `.min()`, `.max()` → `ListOp[Any]`

**What would become static errors (not currently caught):**
- `.field()` on non-Struct column, `.list` on non-List column, `.sum()` on non-numeric list

When ty adds self narrowing support, these can be tightened without changing runtime behavior.
