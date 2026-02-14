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

## Validation

Validate that data conforms to the schema:

```python
df.validate()  # raises SchemaError on mismatch
```

Checks column existence and data types.

## Untyped escape hatch

When you need to drop type safety temporarily:

```python
untyped = df.untyped()                   # UntypedDataFrame
untyped.select("name", "age")            # string-based columns
retyped = untyped.to_typed(Users)        # back to DataFrame[Users]
```
