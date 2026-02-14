# DataFrames

`DataFrame[S]` and `LazyFrame[S]` are the primary interfaces for working with typed data.

## DataFrame vs LazyFrame

| | DataFrame | LazyFrame |
|-|-----------|-----------|
| Execution | Immediate (eager) | Deferred (lazy) |
| `head()`, `tail()`, `sample()` | Available | Not available |
| `collect()` | N/A | Materializes to DataFrame |
| Best for | Interactive work, small data | Optimized pipelines, large data |

Convert between them:

```python
lazy = df.lazy()          # DataFrame → LazyFrame
eager = lazy.collect()    # LazyFrame → DataFrame
```

## Schema-preserving operations

These operations return the same schema type (`DataFrame[S]` → `DataFrame[S]`):

```python
df.filter(Users.age > 25)                    # filter rows
df.sort(Users.score.desc())                  # sort rows
df.sort(Users.name, Users.age)               # sort by multiple columns
df.limit(100)                                # first n rows
df.head(10)                                  # first n rows (eager only)
df.tail(10)                                  # last n rows (eager only)
df.sample(50)                                # random sample (eager only)
df.unique(Users.name)                        # deduplicate by columns
df.drop_nulls(Users.age, Users.score)        # drop null rows
df.with_columns(                             # add/overwrite columns
    (Users.score * 2).alias(Users.score)
)
```

## Schema-transforming operations

These change the column set and return `DataFrame[Any]`:

```python
# select — choose columns
selected = df.select(Users.name, Users.score)  # DataFrame[Any]

# group_by + agg — aggregate
grouped = df.group_by(Users.name).agg(
    Users.score.mean().alias(Users.score)
)  # DataFrame[Any]
```

After a schema-transforming operation, use `cast_schema()` to bind to a named output schema.

## cast_schema

`cast_schema` binds data to a new schema by resolving column mappings:

```python
class UserSummary(Schema):
    name: Column[Utf8]
    score: Column[Float64]

summary = df.select(Users.name, Users.score).cast_schema(UserSummary)
# summary is DataFrame[UserSummary]
```

Resolution precedence per target column:

1. **Explicit mapping** — `mapping={Target.col: Source.col}`
2. **mapped_from** — `col: Column[T] = mapped_from(Source.col)`
3. **Name matching** — target column name matches source column name

The `extra` parameter controls extra columns in the source:

- `extra="drop"` (default) — silently drop extra columns
- `extra="forbid"` — raise `SchemaError` if extra columns exist

## Group by

`group_by()` is available on `DataFrame[S]` and `LazyFrame[S]` — but **not** on `JoinedDataFrame` or `JoinedLazyFrame`. If you need to aggregate joined data, first `cast_schema()` to flatten to a single schema:

```python
# Join → cast_schema → group_by → cast_schema
totals = (
    users.join(orders, on=Users.id == Orders.user_id)
    .cast_schema(UserOrders)
    .group_by(UserOrders.user_name)
    .agg(UserOrders.amount.sum().alias(UserOrders.amount))
    .cast_schema(UserTotals)
)
```

## Introspection

DataFrame provides properties for inspecting dimensions:

```python
df.height      # number of rows (int)
len(df)        # same as height
df.width       # number of columns (int)
df.shape       # (rows, columns) tuple
df.is_empty()  # True if zero rows
```

| Property/Method | DataFrame | LazyFrame | JoinedDataFrame |
|----------------|-----------|-----------|-----------------|
| `height` | Yes | No (requires materialization) | No (cast_schema first) |
| `len()` | Yes | No | No |
| `width` | Yes | Yes (from schema) | No |
| `shape` | Yes | No | No |
| `is_empty()` | Yes | No | No |

`width` raises `TypeError` on `DataFrame[Any]` (schema erased) — use `cast_schema()` first.

## Typed row iteration

`iter_rows_as(row_type)` iterates rows as typed Python objects:

```python
# Using Schema.Row (frozen dataclass)
for row in df.iter_rows_as(Users.Row):
    print(row.name, row.age)  # typed attribute access

# Using dict
for row in df.iter_rows_as(dict):
    print(row["name"], row["age"])
```

`iter_rows_as` accepts any callable that takes `**kwargs`:

- `Schema.Row` — frozen dataclass with typed attributes (recommended)
- `dict` — plain dictionary
- Custom dataclasses, `NamedTuple`, Pydantic models, etc.

`iter_rows_as` is only available on `DataFrame` — not `LazyFrame` (would require materialization) and not `JoinedDataFrame` (use `cast_schema()` first).

## Validation

Validate that data conforms to the schema:

```python
df.validate()  # raises SchemaError on mismatch
```

Checks column existence and data types. See [Validation](validation.md) for the global toggle and auto-validation at IO boundaries.

## What Colnade validates

Colnade provides type safety at two levels:

### Static analysis (ty, pyright, mypy)

- **Schema-aware return types** — `df.filter(...)` returns `DataFrame[Users]`, not just `DataFrame`
- **Type boundary enforcement** — `JoinedDataFrame[S, S2]` is a distinct type from `DataFrame[S]`. You cannot pass a joined frame where a `DataFrame` is expected — you must `cast_schema()` first
- **Schema-transforming operations** — `select()` and `group_by().agg()` return `DataFrame[Any]`, requiring `cast_schema()` to regain a named schema
- **Join conditions** — cross-schema `==` returns `JoinCondition`, same-schema `==` returns `BinOp[Bool]`

### Runtime validation (df.validate())

- **Column existence** — missing columns raise `SchemaError`
- **Data types** — type mismatches raise `SchemaError`
- **Null violations** — non-nullable columns with null values raise `SchemaError`
- **Extra columns** — optionally flagged via `extra="forbid"` on `cast_schema()`

### Current limitations

Column type parameters carry the data type (`Column[UInt64]`) but not the schema they belong to. This means the type checker cannot statically verify that `df.filter(Orders.amount > 5)` is invalid when `df` is a `DataFrame[Users]`. This limitation exists because Python 3.10 lacks `TypeVar` defaults (PEP 696). Schema enforcement at the column level would require `Column[DType, Schema]`, which is planned for future versions.

## Untyped escape hatch

When you need to drop type safety temporarily:

```python
untyped = df.untyped()                   # UntypedDataFrame
untyped.select("name", "age")            # string-based columns
retyped = untyped.to_typed(Users)        # back to DataFrame[Users]
```
