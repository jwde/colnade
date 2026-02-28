# Quick Start

This guide walks through Colnade's core workflow in 5 minutes.

## 1. Define a schema

Schemas declare the shape of your data with typed columns:

```python
import colnade as cn

class Users(cn.Schema):
    id: cn.Column[cn.UInt64]
    name: cn.Column[cn.Utf8]
    age: cn.Column[cn.UInt64]
    score: cn.Column[cn.Float64]
```

Each `Column[DType]` annotation creates a typed descriptor. `Users.age` is a `Column[UInt64]` that the type checker can verify.

## 2. Read typed data

```python
from colnade_polars import read_parquet

df = read_parquet("users.parquet", Users)
# df is DataFrame[Users] — the type checker knows the schema
```

The `read_parquet` function returns a `DataFrame[Users]` with the Polars backend attached. When [validation is enabled](../user-guide/dataframes.md#validation), it also checks that the data matches the schema.

## 3. Transform with type safety

```python
# Filter — column references are attributes, not strings
adults = df.filter(Users.age >= 30)

# Sort — with typed sort expressions
by_score = df.sort(Users.score.desc())

# Compute new values
doubled = df.with_columns((Users.score * 2).alias(Users.score))
```

All these operations return `DataFrame[Users]` — the schema type is preserved.

## 4. Select and bind to an output schema

Operations like `filter` and `sort` keep all columns, so they preserve `DataFrame[Users]`. But `select` changes which columns exist, so it returns `DataFrame[Any]`. Use `cast_schema()` to bind the result to a new schema:

```python
class UserSummary(cn.Schema):
    name: cn.Column[cn.Utf8]
    score: cn.Column[cn.Float64]

summary = df.select(Users.name, Users.score).cast_schema(UserSummary)
# summary is DataFrame[UserSummary]
```

## 5. Write results

```python
from colnade_polars import write_parquet

write_parquet(summary, "summary.parquet")
```

## What the type checker catches

If you misspell a column name:

```python
df.filter(Users.naem > 25)
#         ^^^^^^^^^^
# ty error: Class `Users` has no attribute `naem`
```

If you pass the wrong schema type:

```python
def process_orders(df: DataFrame[Orders]) -> None: ...

process_orders(users_df)  # DataFrame[Users] ≠ DataFrame[Orders]
# ty error: Object of type `DataFrame[Users]` is not assignable to `DataFrame[Orders]`
```

## Next steps

- [User Guide](../user-guide/core-concepts.md) — understand the architecture
- [Tutorials](../tutorials/basic-usage.md) — worked examples with real data
- [API Reference](../api/index.md) — complete API documentation
