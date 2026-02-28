# Schemas

Schemas are the foundation of Colnade's type safety. They declare the structure of your data as Python classes.

## Defining a schema

```python
import colnade as cn

class Users(cn.Schema):
    id: cn.Column[cn.UInt64]
    name: cn.Column[cn.Utf8]
    age: cn.Column[cn.UInt64]
    score: cn.Column[cn.Float64]
```

Each annotation creates a `Column` descriptor on the class. After class creation:

- `Users.id` is a `Column[UInt64]` instance with `name="id"`
- `Users._columns` is a dict: `{"id": Column, "name": Column, "age": Column, "score": Column}`

## Data types

Colnade provides types that map to backend-native types:

| Category | Types |
|----------|-------|
| Boolean | `Bool` |
| Unsigned integers | `UInt8`, `UInt16`, `UInt32`, `UInt64` |
| Signed integers | `Int8`, `Int16`, `Int32`, `Int64` |
| Floating point | `Float32`, `Float64` |
| String / Binary | `Utf8`, `Binary` |
| Temporal | `Date`, `Time`, `Datetime`, `Duration` |
| Nested | `Struct[S]`, `List[T]` |

## Nullable columns

Use `T | None` to mark a column as nullable:

```python
class Users(cn.Schema):
    age: cn.Column[cn.UInt64 | None]    # nullable integer
    tags: cn.Column[cn.List[cn.Utf8] | None]  # nullable list
```

## Schema inheritance

Schemas support standard Python inheritance:

```python
class BaseRecord(cn.Schema):
    id: cn.Column[cn.UInt64]
    created_at: cn.Column[cn.Datetime]

class Users(BaseRecord):
    name: cn.Column[cn.Utf8]
    # Inherits id and created_at
```

## Trait composition

Combine multiple schemas via multiple inheritance:

```python
class Timestamped(cn.Schema):
    created_at: cn.Column[cn.Datetime]
    updated_at: cn.Column[cn.Datetime]

class SoftDeletable(cn.Schema):
    deleted_at: cn.Column[cn.Datetime | None]

class Users(Timestamped, SoftDeletable):
    id: cn.Column[cn.UInt64]
    name: cn.Column[cn.Utf8]
    # Has: id, name, created_at, updated_at, deleted_at
```

## mapped_from

Use `mapped_from` to declare how columns map between schemas during `cast_schema`:

```python
class UserSummary(cn.Schema):
    user_name: cn.Column[cn.Utf8] = cn.mapped_from(Users.name)
    user_id: cn.Column[cn.UInt64] = cn.mapped_from(Users.id)
```

When you call `df.cast_schema(UserSummary)`, the `user_name` column is populated from `Users.name` and `user_id` from `Users.id`.

!!! note "Nullability checking"
    `mapped_from` preserves the source column's type. Mapping a nullable column (`Column[UInt64 | None]`) to a non-nullable annotation (`Column[UInt64]`) is a type error caught by the type checker.

## Value-level constraints with Field()

`Field()` adds domain invariants to columns. These are checked by `df.validate()` or automatically at the `FULL` validation level:

```python
import colnade as cn

class Users(cn.Schema):
    id: cn.Column[cn.UInt64] = cn.Field(unique=True)
    age: cn.Column[cn.UInt64] = cn.Field(ge=0, le=150)
    email: cn.Column[cn.Utf8] = cn.Field(pattern=r"^[^@]+@[^@]+\.[^@]+$")
    status: cn.Column[cn.Utf8] = cn.Field(isin=["active", "inactive"])
    score: cn.Column[cn.Float64] = cn.Field(ge=0.0, le=100.0)
```

Available constraints:

| Constraint | Types | Description |
|-----------|-------|-------------|
| `ge` | numeric, temporal | Greater than or equal |
| `gt` | numeric, temporal | Strictly greater than |
| `le` | numeric, temporal | Less than or equal |
| `lt` | numeric, temporal | Strictly less than |
| `min_length` | string | Minimum string length |
| `max_length` | string | Maximum string length |
| `pattern` | string | Regex pattern match |
| `unique` | any | No duplicate values |
| `isin` | any | Value must be in allowed set |

`Field()` is a superset of `mapped_from()` — use `Field(mapped_from=Source.col, ge=0)` to combine constraints with column mapping.

Constraints are inherited by schema subclasses and can be overridden:

```python
class AdminUsers(Users):
    age: cn.Column[cn.UInt64] = cn.Field(ge=18, le=150)  # tighter lower bound
```

### Cross-column checks with @schema_check

`@schema_check` defines constraints that span multiple columns:

```python
class Events(cn.Schema):
    start: cn.Column[cn.Datetime]
    end: cn.Column[cn.Datetime]

    @cn.schema_check
    def end_after_start(cls):
        return Events.end >= Events.start
```

`@schema_check` methods are inherited by subclasses.

See [DataFrames: Value-level constraints](dataframes.md#level-3-on-your-data-values-value-level-constraints) for validation details.

## Schema.Row

Each schema with at least one column automatically generates a frozen dataclass called `Row` for typed row access:

```python
class Users(cn.Schema):
    id: cn.Column[cn.UInt64]
    name: cn.Column[cn.Utf8]
    age: cn.Column[cn.UInt64]

# Users.Row is a frozen dataclass:
row = Users.Row(id=1, name="Alice", age=30)
row.id    # 1 (int)
row.name  # "Alice" (str)
```

### DType to Python type mapping

| DType | Python type |
|-------|-------------|
| `Bool` | `bool` |
| `UInt8`, `UInt16`, `UInt32`, `UInt64` | `int` |
| `Int8`, `Int16`, `Int32`, `Int64` | `int` |
| `Float32`, `Float64` | `float` |
| `Utf8` | `str` |
| `Binary` | `bytes` |
| `Date` | `datetime.date` |
| `Time` | `datetime.time` |
| `Datetime` | `datetime.datetime` |
| `Duration` | `datetime.timedelta` |
| `List[T]` | `list` |
| `Struct[S]` | `dict` |

Nullable columns (`Column[UInt64 | None]`) produce `int | None` fields.

### Properties

- Row classes are **frozen** (immutable) and use **slots** for memory efficiency
- Class name follows the pattern `"{SchemaName}Row"` (e.g., `UsersRow`)
- Inherited schemas include all parent columns in their Row
- Empty schemas (no columns) do not generate a Row

### Usage with iter_rows_as

`Schema.Row` is designed for use with `DataFrame.iter_rows_as()` — see [DataFrames](dataframes.md#typed-row-iteration).

## SchemaError

Schema validation raises `SchemaError` with structured information:

```python
try:
    df.validate()
except cn.SchemaError as e:
    # Structural violations
    print(e.missing_columns)   # columns in schema but not in data
    print(e.type_mismatches)   # {column: (expected, actual)}
    print(e.extra_columns)     # columns in data but not in schema
    # Value violations (from Field() and @schema_check)
    print(e.value_violations)  # list of ValueViolation objects
```

Each `ValueViolation` contains the column name, constraint description, violation count, and up to 5 sample values.
