# Colnade

A statically type-safe DataFrame abstraction layer for Python.

Colnade replaces string-based column references (`pl.col("age")`) with typed descriptors (`Users.age`), so column misspellings, type mismatches, and schema violations are caught by your type checker — before your code runs.

Works with [ty](https://github.com/astral-sh/ty), mypy, and pyright. No plugins, no code generation.

## Installation

```bash
pip install colnade colnade-polars
```

Colnade requires Python 3.10+. The `colnade-polars` package provides the Polars backend adapter.

## Quick Start

### 1. Define a schema

```python
from colnade import Column, Schema, UInt64, Float64, Utf8

class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]
```

### 2. Read typed data

```python
from colnade_polars import read_parquet

df = read_parquet("users.parquet", Users)
# df is DataFrame[Users] — the type checker knows the schema
```

### 3. Transform with full type safety

```python
# Column references are attributes, not strings
result = (
    df.filter(Users.age > 25)
      .sort(Users.score.desc())
      .select(Users.name, Users.score)
)
```

### 4. Bind to an output schema

```python
class UserSummary(Schema):
    name: Column[Utf8]
    score: Column[Float64]

output = result.cast_schema(UserSummary)
# output is DataFrame[UserSummary]
```

## Key Features

### Type-safe column references

Column references are class attributes verified by the type checker at lint time:

```python
Users.name   # Column[Utf8] — valid
Users.naem   # ty error: Class `Users` has no attribute `naem`
```

### Schema-preserving operations

Operations that don't change the schema (filter, sort, limit, with_columns) preserve the type parameter:

```python
def process(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.filter(Users.age > 25).sort(Users.score.desc())
```

### Typed expressions

Column descriptors build an expression tree with typed operators:

```python
Users.age > 18          # Expr[Bool] — comparison
Users.score * 2         # Expr[Float64] — arithmetic
(Users.age > 18) & (Users.score > 80)  # Expr[Bool] — logical
Users.name.str_starts_with("A")        # Expr[Bool] — string method
```

### Aggregations

```python
result = df.group_by(Users.name).agg(
    Users.score.mean().alias(UserStats.avg_score),
    Users.id.count().alias(UserStats.user_count),
)
```

### Null handling

```python
# Fill nulls, filter nulls, check nulls
df.with_columns(Users.score.fill_null(0.0).alias(Users.score))
df.filter(Users.score.is_not_null())
df.drop_nulls(Users.score)
```

### Joins with typed output

```python
joined = users.join(orders, on=Users.id == Orders.user_id)
# JoinedDataFrame[Users, Orders] — both schemas accessible

class UserOrders(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    amount: Column[Float64]

result = joined.cast_schema(UserOrders)
```

### Schema-polymorphic utility functions

Write generic functions that work with any schema:

```python
from colnade.schema import S

def first_n(df: DataFrame[S], n: int) -> DataFrame[S]:
    return df.head(n)

# Works with any schema — type preserved
users_subset: DataFrame[Users] = first_n(users_df, 10)
```

### Struct and List support

```python
class Address(Schema):
    city: Column[Utf8]
    zip_code: Column[Utf8]

class UserProfile(Schema):
    name: Column[Utf8]
    address: Column[Struct[Address]]
    tags: Column[List[Utf8]]

# Access nested data
df.filter(UserProfile.address.field(Address.city) == "New York")
df.with_columns(UserProfile.tags.list.len().alias(tag_count_col))
```

### Lazy execution

```python
from colnade_polars import scan_parquet

lazy = scan_parquet("users.parquet", Users)
# LazyFrame[Users] — builds a query plan

result = lazy.filter(Users.age > 25).sort(Users.score.desc()).collect()
# Executes the optimized query plan
```

### Untyped escape hatch

When you need to drop down to untyped operations:

```python
untyped = df.untyped()  # UntypedDataFrame — string-based columns
retyped = untyped.to_typed(Users)  # Back to DataFrame[Users]
```

## Type Checker Error Showcase

Colnade catches real errors at lint time. Here are actual error messages from `ty`:

### Misspelled column name

```python
x = Users.agee
```
```
error[unresolved-attribute]: Class `Users` has no attribute `agee`
```

### Schema mismatch at function boundary

```python
df: DataFrame[Users] = read_parquet("users.parquet", Users)
wrong: DataFrame[Orders] = df
```
```
error[invalid-assignment]: Object of type `DataFrame[Users]` is not assignable
to `DataFrame[Orders]`
```

### Nullability mismatch in mapped_from

```python
class Bad(Schema):
    age: Column[UInt8] = mapped_from(Users.age)  # Users.age is Column[UInt8 | None]
```
```
error[invalid-assignment]: Object of type `Column[UInt8 | None]` is not
assignable to `Column[UInt8]`
```

## Comparison with Existing Solutions

| Feature | Colnade | Pandera | StaticFrame | Patito |
|---------|---------|---------|-------------|--------|
| Column refs checked statically | Yes | No | No | No |
| Schema preserved through ops | Yes | Nominal only | No | No |
| Works with existing engines | Yes | Yes | No | Polars only |
| No plugins or code gen | Yes | No (mypy plugin) | Yes | Yes |
| Generic utility functions | Yes | No | No | No |
| Struct/List typed access | Yes | No | No | No |

## Documentation

Full documentation is available at [colnade.com](https://colnade.com/), including:

- [Getting Started](https://colnade.com/getting-started/installation/) — installation and quick start
- [User Guide](https://colnade.com/user-guide/core-concepts/) — concepts, schemas, expressions, joins
- [Tutorials](https://colnade.com/tutorials/basic-usage/) — worked examples with real data
- [API Reference](https://colnade.com/api/) — auto-generated from source

## Examples

Runnable examples are in the [`examples/`](examples/) directory:

- [`basic_usage.py`](examples/basic_usage.py) — Schema definition, filter, select, aggregate
- [`null_handling.py`](examples/null_handling.py) — Nullable columns, fill_null, drop_nulls
- [`joins.py`](examples/joins.py) — Joining DataFrames, JoinedDataFrame, cast_schema
- [`generic_functions.py`](examples/generic_functions.py) — Schema-polymorphic utility functions
- [`nested_types.py`](examples/nested_types.py) — Struct and List column operations
- [`full_pipeline.py`](examples/full_pipeline.py) — Complete ETL pipeline example

## License

MIT
