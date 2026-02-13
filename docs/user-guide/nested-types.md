# Nested Types

Colnade supports struct and list columns with typed access to nested data.

## Struct columns

Define a struct column by parameterizing `Struct` with a schema:

```python
from colnade import Column, Schema, Struct, Utf8

class Address(Schema):
    city: Column[Utf8]
    zip_code: Column[Utf8]

class Users(Schema):
    name: Column[Utf8]
    address: Column[Struct[Address]]
```

### Accessing struct fields

Use `.field()` with a column from the struct's schema:

```python
# Filter by struct field
df.filter(Users.address.field(Address.city) == "New York")

# Check not null
df.filter(Users.address.field(Address.city).is_not_null())
```

The `.field()` method returns a `StructFieldAccess` expression node that supports comparisons and null checks.

## List columns

Define a list column by parameterizing `List` with an element type:

```python
from colnade import Column, Schema, List, Float64, Utf8

class Users(Schema):
    tags: Column[List[Utf8]]
    scores: Column[List[Float64]]
```

### List operations

Access list methods via the `.list` property:

```python
# Length of each list
Users.tags.list.len()

# Get element by index
Users.tags.list.get(0)

# Check if list contains a value
Users.tags.list.contains("python")

# Aggregate list elements
Users.scores.list.sum()
Users.scores.list.mean()
Users.scores.list.min()
Users.scores.list.max()
```

### Using in operations

```python
# Filter by list content
df.filter(Users.tags.list.contains("python"))

# Compute tag counts
df.with_columns(Users.tags.list.len().alias(tag_count_col))

# Sum scores per row
df.with_columns(Users.scores.list.sum().alias(total_score_col))
```

## Nullable nested types

Mark nested columns as nullable with `| None`:

```python
class Users(Schema):
    address: Column[Struct[Address] | None]   # nullable struct
    tags: Column[List[Utf8] | None]           # nullable list
```
