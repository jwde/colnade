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

df = pl.read_parquet("users.parquet")

# Typo — silently produces zero rows
df.filter(pl.col("naem") == "Alice")

# Wrong method on column — crashes at runtime
df.select(pl.col("name").sum())
```

</div>
<div class="compare-col" markdown>

### With Colnade

```python
from colnade_polars import read_parquet
from myapp.schemas import Users

df = read_parquet("users.parquet", Users)

# Typo — caught by your type checker
df.filter(Users.naem == "Alice")  # error!

# Wrong method on column — caught by your type checker
df.select(Users.name.sum())  # error! sum() requires numeric
```

</div>
</div>

---

<div class="feature-grid" markdown>
<div class="feature-card" markdown>

### Type-safe columns

Column references are class attributes. `Users.naem` is a type error. `.sum()` on a string column is a type error. Caught in your editor, not in production.

</div>
<div class="feature-card" markdown>

### Schema preserved

`filter`, `sort`, `with_columns` return `DataFrame[S]`. The schema flows through your entire pipeline.

</div>
<div class="feature-card" markdown>

### Static + runtime safety

Your type checker catches wrong columns and schema mismatches. Runtime validation catches wrong *data* — missing columns, unexpected dtypes, or out-of-range values.

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

Works with `ty`, `mypy`, and `pyright` out of the box. Standard Python classes, nothing extra to install.

</div>
</div>

---

## Quick Example

```python
from colnade import Column, Schema, UInt64, Float64, Utf8
from colnade_polars import read_parquet

class Users(Schema):
    id:    Column[UInt64]
    name:  Column[Utf8]
    age:   Column[UInt64]
    score: Column[Float64]

df = read_parquet("users.parquet", Users)  # DataFrame[Users]

result = (
    df.filter(Users.age > 25)
      .sort(Users.score.desc())
)  # DataFrame[Users] — schema preserved through the pipeline
```

---

**Install:**

```
pip install colnade colnade-polars
```

Or with Pandas: `pip install colnade colnade-pandas` — or Dask: `pip install colnade colnade-dask`

---

<p style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
<a href="getting-started/installation/">Getting Started</a> &middot;
<a href="user-guide/core-concepts/">User Guide</a> &middot;
<a href="tutorials/basic-usage/">Tutorials</a> &middot;
<a href="api/">API Reference</a> &middot;
<a href="comparison/">Comparison</a>
</p>
