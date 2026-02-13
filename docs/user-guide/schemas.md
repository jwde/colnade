# Schemas

Schemas are the foundation of Colnade's type safety. They declare the structure of your data as Python classes.

## Defining a schema

```python
from colnade import Column, Schema, UInt64, Float64, Utf8

class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]
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
class Users(Schema):
    age: Column[UInt64 | None]    # nullable integer
    tags: Column[List[Utf8] | None]  # nullable list
```

## Schema inheritance

Schemas support standard Python inheritance:

```python
class BaseRecord(Schema):
    id: Column[UInt64]
    created_at: Column[Datetime]

class Users(BaseRecord):
    name: Column[Utf8]
    # Inherits id and created_at
```

## Trait composition

Combine multiple schemas via multiple inheritance:

```python
class Timestamped(Schema):
    created_at: Column[Datetime]
    updated_at: Column[Datetime]

class SoftDeletable(Schema):
    deleted_at: Column[Datetime | None]

class Users(Timestamped, SoftDeletable):
    id: Column[UInt64]
    name: Column[Utf8]
    # Has: id, name, created_at, updated_at, deleted_at
```

## mapped_from

Use `mapped_from` to declare how columns map between schemas during `cast_schema`:

```python
from colnade import mapped_from

class UserSummary(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    user_id: Column[UInt64] = mapped_from(Users.id)
```

When you call `df.cast_schema(UserSummary)`, the `user_name` column is populated from `Users.name` and `user_id` from `Users.id`.

!!! note "Nullability checking"
    `mapped_from` preserves the source column's type. Mapping a nullable column (`Column[UInt64 | None]`) to a non-nullable annotation (`Column[UInt64]`) is a type error caught by the type checker.

## SchemaError

Schema validation raises `SchemaError` with structured information:

```python
from colnade import SchemaError

try:
    df.validate()
except SchemaError as e:
    print(e.missing_columns)   # columns in schema but not in data
    print(e.type_mismatches)   # {column: (expected, actual)}
    print(e.extra_columns)     # columns in data but not in schema
```
