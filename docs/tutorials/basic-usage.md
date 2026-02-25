# Basic Usage

This tutorial demonstrates the core Colnade workflow: define a schema, read data, filter, sort, compute, and output.

!!! tip "Runnable example"
    The complete code for this tutorial is in [`examples/basic_usage.py`](https://github.com/jwde/colnade/blob/main/examples/basic_usage.py).

## Define schemas

```python
from colnade import Column, Schema, UInt64, Float64, Utf8

class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]

class UserSummary(Schema):
    name: Column[Utf8]
    score: Column[Float64]
```

## Read typed data

```python
from colnade_polars import read_parquet

df = read_parquet("users.parquet", Users)
# df is DataFrame[Users] — the type checker knows the schema
```

`read_parquet` returns a `DataFrame[Users]` with the Polars backend attached. When [validation is enabled](../user-guide/dataframes.md#validation), it also checks columns and types against the schema.

## Filter rows

```python
adults = df.filter(Users.age >= 30)
```

`Users.age >= 30` produces `Expr[Bool]`. The `filter` method passes this expression to the Polars backend, which translates it to `pl.col("age") >= 30`. The result is `DataFrame[Users]` — schema preserved.

## Sort

```python
by_score = df.sort(Users.score.desc())
```

`.desc()` creates a `SortExpr` that tells the backend to sort in descending order.

## Compute new values

```python
doubled = df.with_columns((Users.score * 2).alias(Users.score))
```

`(Users.score * 2)` builds a `BinOp[Float64]`. `.alias(Users.score)` wraps it in an `AliasedExpr` targeting the `score` column. The result overwrites the score column with doubled values.

## Select and cast

```python
summary = df.select(Users.name, Users.score).cast_schema(UserSummary)
```

`select` returns `DataFrame[Any]` (the column set changed). `cast_schema(UserSummary)` resolves column names and binds to the target schema. Since `UserSummary.name` matches `Users.name` by name, no `mapped_from` is needed.

## Write results

```python
from colnade_polars import write_parquet

write_parquet(summary, "summary.parquet")
```
