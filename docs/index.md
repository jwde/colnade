---
title: Colnade
template: home.html
hide:
  - navigation
  - toc
---

<div class="compare-grid" markdown>
<div class="compare-col" markdown>

### Without type safety

```python
import polars as pl

# Typo silently produces wrong results
df.filter(pl.col("naem") == "Alice")

# Wrong column from wrong table — no error
df.select(pl.col("amount"))  # amount is on orders, not users
```

</div>
<div class="compare-col" markdown>

### With Colnade

```python
from colnade_polars import read_parquet

# Typo caught by type checker instantly
df.filter(Users.naem == "Alice")  # error: no attribute 'naem'

# Schema mismatch caught at function boundaries
process_orders(users_df)  # error: expected DataFrame[Orders]
```

</div>
</div>

---

<div class="feature-grid" markdown>
<div class="feature-card" markdown>

### Type-safe columns

`Users.naem` is a type error, not a runtime crash. Column references are class attributes checked by your editor.

</div>
<div class="feature-card" markdown>

### Schema preserved

`filter`, `sort`, `with_columns` return `DataFrame[S]`. The type parameter flows through your entire pipeline.

</div>
<div class="feature-card" markdown>

### Static + runtime safety

Your type checker catches wrong columns and schema mismatches in your code. Runtime validation catches wrong *data* — files with missing columns, unexpected dtypes, or values out of range.

</div>
<div class="feature-card" markdown>

### Backend agnostic

Write once, run on **Polars**, **Pandas**, or **Dask**. Same schema, same expressions, same type safety.

</div>
<div class="feature-card" markdown>

### Generic functions

`def f(df: DataFrame[S]) -> DataFrame[S]` works with any schema. Build reusable utilities without losing type information.

</div>
<div class="feature-card" markdown>

### No plugins or codegen

Works with `ty`, `mypy`, and `pyright` out of the box. Standard Python type annotations, nothing extra to install.

</div>
</div>

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

<p style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
<a href="getting-started/installation/">Installation</a> &middot;
<a href="user-guide/core-concepts/">User Guide</a> &middot;
<a href="tutorials/basic-usage/">Tutorials</a> &middot;
<a href="api/">API Reference</a> &middot;
<a href="comparison/">Comparison</a>
</p>
