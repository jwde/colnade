# DataFrames

`DataFrame[S]` and `LazyFrame[S]` are the primary interfaces for working with typed data.

## Constructing DataFrames

Every backend provides `from_rows()` and `from_dict()` for creating typed DataFrames from Python data. The schema drives dtype coercion — you never need to specify backend-specific types.

### From rows

```python
from colnade_polars import from_rows

df = from_rows(Users, [
    Users.Row(id=1, name="Alice", age=30, score=85.0),
    Users.Row(id=2, name="Bob", age=25, score=92.5),
])
# df is DataFrame[Users] with correct dtypes
```

`from_rows` accepts `Row[S]` instances — the type checker verifies that rows match the schema, so passing `Orders.Row` where `Users.Row` is expected is a static error. For row-oriented dicts, construct `Row` instances first: `Users.Row(**d)`.

### From columnar dict

```python
from colnade_polars import from_dict

df = from_dict(Users, {
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [30, 25, 35],
    "score": [85.0, 92.5, 78.0],
})
```

Both functions validate the data if validation is enabled (see [Validation](#validation)).

### From files

Use `read_parquet()`, `read_csv()`, or their lazy equivalents (`scan_parquet()`, `scan_csv()`):

```python
from colnade_polars import read_parquet

df = read_parquet("users.parquet", Users)
```

## DataFrame vs LazyFrame

| | DataFrame | LazyFrame |
|-|-----------|-----------|
| Execution | Immediate (eager) | Deferred (lazy) |
| `head()`, `tail()` | Available | Available |
| `height`, `len()` | Available | Available (triggers computation) |
| `to_batches()` | Available | Available (triggers computation) |
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

# agg — aggregate all rows into a single row
summary = df.agg(
    Users.score.mean().alias(Stats.avg_score),
    Users.id.count().alias(Stats.user_count),
)  # DataFrame[Any]

# group_by + agg — grouped aggregation
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

### cast_schema is a trust boundary

`cast_schema` is analogous to a type cast in Go or Rust — it asserts that the data conforms to the target schema. The type checker verifies that expressions reference valid columns on the *input* schema, but `cast_schema` is a promise about the *output*. If you `.select()` the wrong columns, the type checker won't catch it.

**Mitigations:**

- **Use `mapped_from`** on output schema fields to create static links between input and output columns. The more fields that declare their provenance, the narrower the trust gap:

    ```python
    class UserRevenue(Schema):
        user_name: Column[Utf8] = mapped_from(Users.name)
        user_id: Column[UInt64] = mapped_from(Users.id)
        total_amount: Column[Float64]  # only this field is "trust me"
    ```

- **Use `extra="forbid"`** to catch unexpected columns that might indicate a wrong select.
- **Enable validation** — with validation on, `df.validate()` after `cast_schema` verifies structural conformance at runtime. Consider calling `.cast_schema(Target).validate()` at critical pipeline boundaries.

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
| `height` | Yes | Available (triggers computation) | No (cast_schema first) |
| `len()` | Yes | Available (triggers computation) | No |
| `width` | Yes | Yes (from schema) | No |
| `shape` | Yes | Available (triggers computation) | No |
| `is_empty()` | Yes | Available (triggers computation) | No |

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

Checks column existence, data types, and nullability constraints.

### Validation levels

| Level | Behavior |
|-------|----------|
| `ValidationLevel.OFF` | No runtime checks. Trust the type checker. Zero overhead. (default) |
| `ValidationLevel.STRUCTURAL` | Check columns exist, dtypes match, nullability. Also checks literal type compatibility in expressions. |
| `ValidationLevel.FULL` | Structural checks plus value-level constraints from `Field()` metadata. |

Enable auto-validation at data boundaries:

```python
from colnade import ValidationLevel

colnade.set_validation(ValidationLevel.STRUCTURAL)  # or FULL
# Strings and booleans still work for convenience:
colnade.set_validation("structural")
colnade.set_validation(True)  # → STRUCTURAL
```

Or via environment variable:

```bash
COLNADE_VALIDATE=structural pytest tests/
COLNADE_VALIDATE=full pytest tests/
# Legacy: COLNADE_VALIDATE=1 → STRUCTURAL
```

`df.validate()` always runs the full level of checks regardless of the toggle — calling it explicitly signals intent.

## What Colnade validates

Colnade catches errors at three levels (see also [Core Concepts: Safety Model](core-concepts.md#the-safety-model)):

### Level 1: In your editor (static analysis)

Your type checker (`ty`, `pyright`, `mypy`) catches errors before code runs:

- **Schema-aware return types** — `df.filter(...)` returns `DataFrame[Users]`, not just `DataFrame`
- **Type boundary enforcement** — `JoinedDataFrame[S, S2]` is a distinct type from `DataFrame[S]`. You cannot pass a joined frame where a `DataFrame` is expected — you must `cast_schema()` first
- **Schema-transforming operations** — `select()` and `group_by().agg()` return `DataFrame[Any]`, requiring `cast_schema()` to regain a named schema
- **Join conditions** — cross-schema `==` returns `JoinCondition`, same-schema `==` returns `BinOp[Bool]`

### Level 2: At data boundaries (runtime structural validation)

When validation is enabled, data boundaries and `df.validate()` check:

- **Column existence** — missing columns raise `SchemaError`
- **Data types** — type mismatches raise `SchemaError`
- **Null violations** — non-nullable columns with null values raise `SchemaError`
- **Extra columns** — optionally flagged via `extra="forbid"` on `cast_schema()`
- **Expression column membership** — operations like `filter`, `sort`, `select` verify that all column references in expressions belong to the frame's schema (e.g., using `Orders.amount` on a `DataFrame[Users]` raises `SchemaError`). On `JoinedDataFrame`, columns from either schema are accepted.

### Level 3: On your data values (value-level constraints)

Value-level constraints validate domain invariants using `Field()` metadata:

```python
from colnade import Column, Schema, UInt64, Utf8, Float64
from colnade.constraints import Field, schema_check

class Users(Schema):
    id: Column[UInt64] = Field(unique=True)
    age: Column[UInt64] = Field(ge=0, le=150)
    name: Column[Utf8] = Field(min_length=1)
    email: Column[Utf8] = Field(pattern=r"^[^@]+@[^@]+\.[^@]+$")
    score: Column[Float64] = Field(ge=0.0, le=100.0)
    status: Column[Utf8] = Field(isin=["active", "inactive"])
```

Available constraints:

| Constraint | Types | Meaning |
|-----------|-------|---------|
| `ge` | Numeric, temporal | Value >= bound |
| `gt` | Numeric, temporal | Value > bound |
| `le` | Numeric, temporal | Value <= bound |
| `lt` | Numeric, temporal | Value < bound |
| `min_length` | String | String length >= n |
| `max_length` | String | String length <= n |
| `pattern` | String | Matches regex |
| `unique` | Any | No duplicate values |
| `isin` | Any | Value in allowed set |

`Field()` is a superset of `mapped_from()` — use `Field(ge=0, mapped_from=Source.age)` to combine constraints with column mapping.

Cross-column constraints use `@schema_check`:

```python
class Events(Schema):
    start: Column[UInt64]
    end: Column[UInt64]

    @schema_check
    def start_before_end(cls):
        return Events.start <= Events.end
```

Value constraints are checked by `df.validate()` (always) and by auto-validation when the level is `"full"`. Structural-level auto-validation skips value checks for performance.

### Current limitations

Column type parameters carry the data type (`Column[UInt64]`) but not the schema they belong to. This means the type checker cannot *statically* verify that `df.filter(Orders.amount > 5)` is invalid when `df` is a `DataFrame[Users]`. This limitation exists because Python 3.10 lacks `TypeVar` defaults (PEP 696). Schema enforcement at the column level would require `Column[DType, Schema]`, which is planned for future versions.

**Runtime mitigation:** When validation is enabled (`STRUCTURAL` or `FULL`), all DataFrame/LazyFrame operations validate expression column membership at runtime. See [Type Checker Integration: Wrong-schema columns](type-checking.md#wrong-schema-columns-in-expressions) for details.

## Adding computed columns

Use `with_columns` to add a computed column, then `cast_schema` to transition to a richer child schema:

```python
class EnrichedUsers(Users):
    risk_score: Column[Float64]

result = df.with_columns(
    (Users.age * 0.1 + Users.score * 0.9).alias(EnrichedUsers.risk_score)
).cast_schema(EnrichedUsers)
# result is DataFrame[EnrichedUsers]
```

This works because `cast_schema` recognizes schema inheritance — columns declared on the child schema (`risk_score`) that aren't in the parent (`Users`) are resolved by identity (the column name matches itself in the data). Columns inherited from the parent resolve by normal name matching.

## Escape hatches

### with_raw — scoped escape (recommended)

When you need to use engine-native operations not exposed by Colnade, `with_raw` lets you operate on the raw DataFrame within a bounded scope — like Rust's `unsafe` block:

```python
# Apply a Polars-native operation, then re-enter the typed world
result = df.with_raw(
    lambda raw: raw.with_columns(
        pl.col("age").map_batches(some_custom_fn)
    )
)
# result is still DataFrame[Users]
# validated automatically if validation is enabled
```

For complex multi-step logic, use a named function:

```python
def custom_transform(raw_df: pl.DataFrame) -> pl.DataFrame:
    # complex engine-native logic here
    return raw_df.with_columns(...)

result = df.with_raw(custom_transform)
```

`with_raw` is available on `DataFrame` and `LazyFrame`, but **not** on `JoinedDataFrame` — use `cast_schema()` first.
