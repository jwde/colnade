# Colnade

**A statically type-safe DataFrame abstraction layer for Python.**

Colnade replaces string-based column references (`pl.col("age")`) with typed descriptors (`Users.age`), so column misspellings, type mismatches, and schema violations are caught by your type checker — before your code runs.

Works with [ty](https://github.com/astral-sh/ty), mypy, and pyright. No plugins, no code generation.

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[Tutorials](tutorials/basic-usage.md){ .md-button }

---

## Quick Example

=== "Define Schema"

    ```python
    from colnade import Column, Schema, UInt64, Float64, Utf8

    class Users(Schema):
        id: Column[UInt64]
        name: Column[Utf8]
        age: Column[UInt64]
        score: Column[Float64]
    ```

=== "Read & Transform"

    ```python
    from colnade_polars import read_parquet

    df = read_parquet("users.parquet", Users)

    result = (
        df.filter(Users.age > 25)
          .sort(Users.score.desc())
          .select(Users.name, Users.score)
    )
    ```

=== "Output Schema"

    ```python
    class UserSummary(Schema):
        name: Column[Utf8]
        score: Column[Float64]

    output = result.cast_schema(UserSummary)
    # output is DataFrame[UserSummary]
    ```

---

## Key Features

- **Type-safe column references** — `Users.naem` is a type error, not a runtime crash
- **Schema-preserving operations** — `filter`, `sort`, `with_columns` preserve `DataFrame[S]`
- **Typed expressions** — `Users.age > 18` produces `Expr[Bool]`, `Users.score * 2` produces `Expr[Float64]`
- **Backend agnostic** — works with Polars today, with adapters for other engines
- **Generic utility functions** — write `def f(df: DataFrame[S]) -> DataFrame[S]` that works with any schema
- **Struct and List support** — typed access to nested data structures
- **No plugins or codegen** — works with standard type checkers out of the box
