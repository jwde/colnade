# Colnade API Reference

## Schema Layer

### `Schema`

Base class for defining typed data schemas. Uses a metaclass (`SchemaMeta`) that
creates `Column` descriptors from class annotations.

```python
from colnade import Column, Schema, UInt64, Float64, Utf8

class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]
```

**Class attribute:** `_columns: dict[str, Column[Any]]` — map of column names to descriptors, populated by the metaclass.

Schema classes support inheritance:

```python
class BaseRecord(Schema):
    id: Column[UInt64]
    created_at: Column[Datetime]

class Users(BaseRecord):
    name: Column[Utf8]
    # Inherits id and created_at from BaseRecord
```

---

### `Column[DType]`

A typed column descriptor. Created automatically by `SchemaMeta` from schema annotations. Provides operators and methods that build expression AST nodes.

**Attributes:**
- `name: str` — column name
- `dtype: Any` — the data type annotation
- `schema: type` — the owning schema class
- `_mapped_from: Column[Any] | None` — source column for `cast_schema` resolution

#### Comparison operators

All return `BinOp[Bool]` (or `JoinCondition` for cross-schema `==`):

| Operator | Example | Result |
|----------|---------|--------|
| `>` | `Users.age > 18` | `BinOp[Bool]` |
| `<` | `Users.age < 65` | `BinOp[Bool]` |
| `>=` | `Users.age >= 18` | `BinOp[Bool]` |
| `<=` | `Users.age <= 65` | `BinOp[Bool]` |
| `==` | `Users.age == 30` | `BinOp[Bool]` |
| `!=` | `Users.age != 0` | `BinOp[Bool]` |
| `==` (cross-schema) | `Users.id == Orders.user_id` | `JoinCondition` |

#### Arithmetic operators

All return `BinOp[DType]`:

| Operator | Example |
|----------|---------|
| `+` | `Users.age + 1` |
| `-` | `Users.age - 1` |
| `*` | `Users.score * 2` |
| `/` | `Users.score / 100` |
| `%` | `Users.age % 10` |
| `-` (unary) | `-Users.score` |

Reverse operators (`1 + Users.age`) are also supported.

#### Aggregation methods

| Method | Return type | Description |
|--------|-------------|-------------|
| `sum()` | `Agg[DType]` | Sum of values |
| `mean()` | `Agg[Float64]` | Mean of values |
| `min()` | `Agg[DType]` | Minimum value |
| `max()` | `Agg[DType]` | Maximum value |
| `count()` | `Agg[UInt32]` | Count of non-null values |
| `std()` | `Agg[Float64]` | Standard deviation |
| `var()` | `Agg[Float64]` | Variance |
| `first()` | `Agg[DType]` | First value |
| `last()` | `Agg[DType]` | Last value |
| `n_unique()` | `Agg[UInt32]` | Count of unique values |

#### String methods (Utf8 columns)

| Method | Return type | Description |
|--------|-------------|-------------|
| `str_contains(pattern)` | `FunctionCall[Bool]` | Contains substring |
| `str_starts_with(prefix)` | `FunctionCall[Bool]` | Starts with prefix |
| `str_ends_with(suffix)` | `FunctionCall[Bool]` | Ends with suffix |
| `str_len()` | `FunctionCall[UInt32]` | String length |
| `str_to_lowercase()` | `FunctionCall[Utf8]` | Convert to lowercase |
| `str_to_uppercase()` | `FunctionCall[Utf8]` | Convert to uppercase |
| `str_strip()` | `FunctionCall[Utf8]` | Strip whitespace |
| `str_replace(pattern, replacement)` | `FunctionCall[Utf8]` | Replace substring |

#### Temporal methods (Datetime columns)

| Method | Return type | Description |
|--------|-------------|-------------|
| `dt_year()` | `FunctionCall[Int32]` | Extract year |
| `dt_month()` | `FunctionCall[Int32]` | Extract month |
| `dt_day()` | `FunctionCall[Int32]` | Extract day |
| `dt_hour()` | `FunctionCall[Int32]` | Extract hour |
| `dt_minute()` | `FunctionCall[Int32]` | Extract minute |
| `dt_second()` | `FunctionCall[Int32]` | Extract second |
| `dt_truncate(interval)` | `FunctionCall[Datetime]` | Truncate to interval |

#### Null handling

| Method | Return type | Description |
|--------|-------------|-------------|
| `is_null()` | `UnaryOp[Bool]` | Check if null |
| `is_not_null()` | `UnaryOp[Bool]` | Check if not null |
| `fill_null(value)` | `FunctionCall[DType]` | Replace nulls with value |
| `assert_non_null()` | `FunctionCall[DType]` | Assert non-null (runtime) |

#### NaN handling (Float columns)

| Method | Return type | Description |
|--------|-------------|-------------|
| `is_nan()` | `UnaryOp[Bool]` | Check if NaN |
| `fill_nan(value)` | `FunctionCall[DType]` | Replace NaN with value |

#### Other methods

| Method | Return type | Description |
|--------|-------------|-------------|
| `alias(target)` | `AliasedExpr[Any]` | Alias to target column |
| `as_column(target)` | `AliasedExpr[Any]` | Alias to target column (synonym) |
| `cast(new_dtype)` | `FunctionCall[Any]` | Cast to new data type |
| `over(*partition_by)` | `FunctionCall[DType]` | Window function |
| `desc()` | `SortExpr` | Sort descending |
| `asc()` | `SortExpr` | Sort ascending |
| `field(col)` | `StructFieldAccess[T]` | Access struct field |
| `list` (property) | `ListAccessor[Any]` | Access list operations |

---

### `mapped_from(source)`

Declares a column mapping for `cast_schema()` resolution:

```python
class UserSummary(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    user_id: Column[UInt64] = mapped_from(Users.id)
```

**Parameters:**
- `source: Column[DType]` — the source column from another schema

**Returns:** `Column[DType]` (a sentinel detected by `SchemaMeta`)

---

### `SchemaError`

Exception raised when data does not conform to the declared schema.

**Attributes:**
- `missing_columns: list[str]` — columns in schema but not in data
- `extra_columns: list[str]` — columns in data but not in schema (with `extra="forbid"`)
- `type_mismatches: dict[str, tuple[str, str]]` — `{column: (expected, actual)}`
- `null_violations: list[str]` — non-nullable columns with null values

---

### `ListAccessor[DType]`

Accessor for list column operations. Created via `Column.list` property.

| Method | Return type | Description |
|--------|-------------|-------------|
| `len()` | `ListOp[UInt32]` | List length |
| `get(index)` | `ListOp[Any]` | Get element by index |
| `contains(value)` | `ListOp[Bool]` | Check if list contains value |
| `sum()` | `ListOp[Any]` | Sum of list elements |
| `mean()` | `ListOp[Any]` | Mean of list elements |
| `min()` | `ListOp[Any]` | Minimum of list elements |
| `max()` | `ListOp[Any]` | Maximum of list elements |

---

## Data Types

### Type categories

| Category | Base class | Used for |
|----------|------------|----------|
| Numeric | `NumericType` | All numeric types |
| Integer | `IntegerType(NumericType)` | Signed and unsigned integers |
| Float | `FloatType(NumericType)` | Floating-point numbers |
| Temporal | `TemporalType` | Date, time, datetime, duration |

### Concrete types

| Type | Category | Description |
|------|----------|-------------|
| `Bool` | — | Boolean |
| `UInt8` | Integer | 8-bit unsigned integer |
| `UInt16` | Integer | 16-bit unsigned integer |
| `UInt32` | Integer | 32-bit unsigned integer |
| `UInt64` | Integer | 64-bit unsigned integer |
| `Int8` | Integer | 8-bit signed integer |
| `Int16` | Integer | 16-bit signed integer |
| `Int32` | Integer | 32-bit signed integer |
| `Int64` | Integer | 64-bit signed integer |
| `Float32` | Float | 32-bit floating point |
| `Float64` | Float | 64-bit floating point |
| `Utf8` | — | UTF-8 string |
| `Binary` | — | Raw binary data |
| `Date` | Temporal | Calendar date |
| `Time` | Temporal | Time of day |
| `Datetime` | Temporal | Date and time |
| `Duration` | Temporal | Time duration |

### Parameterized types

| Type | Description | Example |
|------|-------------|---------|
| `Struct[S]` | Struct parameterized by a Schema | `Column[Struct[Address]]` |
| `List[T]` | List parameterized by element type | `Column[List[Utf8]]` |

Nullable types use `T | None`:

```python
class Users(Schema):
    age: Column[UInt64 | None]       # nullable integer
    tags: Column[List[Utf8] | None]  # nullable list
```

---

## DataFrame Layer

### `DataFrame[S]`

A typed, materialized DataFrame parameterized by a Schema.

#### Schema-preserving operations (return `DataFrame[S]`)

| Method | Parameters | Description |
|--------|------------|-------------|
| `filter(predicate)` | `Expr[Bool]` | Filter rows by boolean expression |
| `sort(*columns, descending=False)` | `Column[Any] \| SortExpr` | Sort rows |
| `limit(n)` | `int` | Limit to first n rows |
| `head(n=5)` | `int` | First n rows |
| `tail(n=5)` | `int` | Last n rows |
| `sample(n)` | `int` | Random sample of n rows |
| `unique(*columns)` | `Column[Any]` | Remove duplicate rows |
| `drop_nulls(*columns)` | `Column[Any]` | Drop rows with nulls |
| `with_columns(*exprs)` | `AliasedExpr \| Expr` | Add/overwrite columns |

#### Schema-transforming operations (return `DataFrame[Any]`)

| Method | Parameters | Description |
|--------|------------|-------------|
| `select(*columns)` | `Column[Any]` | Select columns |
| `group_by(*keys)` | `Column[Any]` | Group for aggregation (returns `GroupBy[S]`) |

#### Schema transition

| Method | Parameters | Description |
|--------|------------|-------------|
| `cast_schema(schema, mapping=None, extra="drop")` | `type[S3]` | Bind to a new schema |

`cast_schema` resolution precedence per target column:
1. Explicit `mapping` dict
2. Target column's `mapped_from` attribute
3. Name matching against source columns

The `extra` parameter controls extra columns in the source:
- `"drop"` (default): silently drop extra columns
- `"forbid"`: raise `SchemaError` if extra columns exist

#### Other

| Method | Returns | Description |
|--------|---------|-------------|
| `lazy()` | `LazyFrame[S]` | Convert to lazy query plan |
| `untyped()` | `UntypedDataFrame` | Drop type information |
| `validate()` | `DataFrame[S]` | Validate data against schema |
| `join(other, on, how="inner")` | `JoinedDataFrame[S, S2]` | Join with another DataFrame |

---

### `LazyFrame[S]`

A typed, lazy query plan. Same operations as `DataFrame` except no `head()`, `tail()`, `sample()` (materialized-only).

Additional method:
- `collect() -> DataFrame[S]` — materialize the query plan

---

### `GroupBy[S]`

Result of `DataFrame.group_by()`.

- `agg(*exprs: AliasedExpr) -> DataFrame[Any]` — aggregate grouped data

---

### `JoinedDataFrame[S, S2]`

Result of joining two DataFrames. Accepts columns from either schema S or S2.

Same schema-preserving operations as `DataFrame`, returning `JoinedDataFrame[S, S2]`.

- `cast_schema(schema)` — flatten to a single-schema `DataFrame[S3]`
- `select(*columns)` — select columns, returns `DataFrame[Any]`

For join columns with the same name in both schemas, use `mapped_from` to disambiguate:

```python
class JoinOutput(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    order_amount: Column[Float64] = mapped_from(Orders.amount)
```

---

### `JoinedLazyFrame[S, S2]`

Lazy equivalent of `JoinedDataFrame`. Same operations except no `head()`, `tail()`, `sample()`.

- `collect() -> JoinedDataFrame[S, S2]` — materialize

---

### `UntypedDataFrame`

String-based escape hatch with no schema parameter.

| Method | Description |
|--------|-------------|
| `select(*columns: str)` | Select columns by name |
| `filter(expr)` | Filter rows |
| `with_columns(*exprs)` | Add/overwrite columns |
| `sort(*columns: str, descending=False)` | Sort rows |
| `limit(n)` | Limit rows |
| `head(n=5)` | First n rows |
| `tail(n=5)` | Last n rows |
| `to_typed(schema)` | Bind to a schema, returns `DataFrame[S]` |

---

### `UntypedLazyFrame`

Lazy equivalent of `UntypedDataFrame`.

- `collect() -> UntypedDataFrame`
- `to_typed(schema) -> LazyFrame[S]`

---

## Expression AST

All expression nodes inherit from `Expr[DType]`.

| Node | Description | Created by |
|------|-------------|------------|
| `ColumnRef[DType]` | Reference to a named column | `Column._ref()` (internal) |
| `Literal[DType]` | A literal value | `lit(value)` or automatic wrapping |
| `BinOp[DType]` | Binary operation | Operators on Column/Expr |
| `UnaryOp[DType]` | Unary operation | `-col`, `is_null()`, `is_not_null()`, `is_nan()` |
| `FunctionCall[DType]` | Named function call | String/temporal/null methods |
| `Agg[DType]` | Aggregation | `.sum()`, `.mean()`, etc. |
| `AliasedExpr[DType]` | Expression with alias target | `.alias(target)` |
| `SortExpr` | Sort expression | `.desc()`, `.asc()` |
| `StructFieldAccess[DType]` | Struct field access | `.field(col)` |
| `ListOp[DType]` | List operation | `.list.len()`, etc. |
| `JoinCondition` | Cross-schema equality | `Users.id == Orders.user_id` |

The `lit()` function creates a `Literal` node:

```python
from colnade import lit
df.filter(Users.age > lit(18))
```

---

## Backend Protocol

### `BackendProtocol`

Interface that all backend adapters must implement. Defined in `colnade._protocols`.

Methods:
- `translate_expr(expr)` — translate expression AST to backend-native
- `filter(source, predicate)` — filter rows
- `sort(source, by, descending)` — sort rows
- `limit(source, n)` / `head(source, n)` / `tail(source, n)` / `sample(source, n)`
- `unique(source, columns)` — deduplicate
- `drop_nulls(source, columns)` — remove null rows
- `select(source, columns)` — project columns
- `with_columns(source, exprs)` — add/overwrite columns
- `group_by_agg(source, keys, aggs)` — grouped aggregation
- `join(left, right, on, how)` — join two datasets
- `cast_schema(source, column_mapping)` — rename/select columns
- `lazy(source)` / `collect(source)` — lazy/eager conversion
- `validate_schema(source, schema)` — validate data against schema

---

## Polars Backend (`colnade-polars`)

### `PolarsBackend`

Reference implementation of `BackendProtocol` for Polars.

```python
from colnade_polars import PolarsBackend
backend = PolarsBackend()
```

### I/O Functions

```python
from colnade_polars import (
    read_parquet,
    scan_parquet,
    read_csv,
    scan_csv,
    write_parquet,
    write_csv,
)
```

| Function | Returns | Description |
|----------|---------|-------------|
| `read_parquet(path, schema)` | `DataFrame[S]` | Read Parquet file |
| `scan_parquet(path, schema)` | `LazyFrame[S]` | Lazily scan Parquet file |
| `read_csv(path, schema, **kwargs)` | `DataFrame[S]` | Read CSV file |
| `scan_csv(path, schema, **kwargs)` | `LazyFrame[S]` | Lazily scan CSV file |
| `write_parquet(df, path)` | `None` | Write DataFrame to Parquet |
| `write_csv(df, path, **kwargs)` | `None` | Write DataFrame to CSV |

### Dtype Mapping

```python
from colnade_polars import map_colnade_dtype, map_polars_dtype
```

| Function | Description |
|----------|-------------|
| `map_colnade_dtype(dtype)` | Convert Colnade dtype to Polars dtype |
| `map_polars_dtype(pl_dtype)` | Convert Polars dtype to Colnade dtype |
