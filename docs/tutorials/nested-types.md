# Nested Types

This tutorial demonstrates working with struct and list columns.

!!! tip "Runnable example"
    The complete code is in [`examples/nested_types.py`](https://github.com/jwde/colnade/blob/main/examples/nested_types.py).

## Define schemas with nested types

```python
from colnade import Column, Schema, Struct, List, UInt64, Float64, Utf8

class Address(Schema):
    city: Column[Utf8]
    zip_code: Column[Utf8]

class UserProfile(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    address: Column[Struct[Address]]
    tags: Column[List[Utf8]]
    scores: Column[List[Float64]]
```

## Struct field access

Access fields within a struct column using `.field()`:

```python
# Filter by struct field value
new_yorkers = df.filter(
    UserProfile.address.field(Address.city) == "New York"
)

# Check if a struct field is not null
df.filter(UserProfile.address.field(Address.zip_code).is_not_null())
```

`.field(Address.city)` creates a `StructFieldAccess` node. The backend translates it to `pl.col("address").struct.field("city")`.

## List operations

Access list methods via the `.list` property:

```python
# Count elements in each list
tag_counts = df.with_columns(
    UserProfile.tags.list.len().alias(UserProfile.tags)
)

# Check if list contains a value
python_users = df.filter(
    UserProfile.tags.list.contains("python")
)

# Get element by index (0-based)
first_tags = df.with_columns(
    UserProfile.tags.list.get(0).alias(UserProfile.tags)
)

# Aggregate list elements (numeric lists)
score_totals = df.with_columns(
    UserProfile.scores.list.sum().alias(UserProfile.scores)
)
```

## Available list methods

| Method | Description | Return type |
|--------|-------------|-------------|
| `.list.len()` | Number of elements | `ListOp[UInt32]` |
| `.list.get(i)` | Element at index | `ListOp[Any]` |
| `.list.contains(v)` | Contains value? | `ListOp[Bool]` |
| `.list.sum()` | Sum of elements | `ListOp[Any]` |
| `.list.mean()` | Mean of elements | `ListOp[Any]` |
| `.list.min()` | Minimum element | `ListOp[Any]` |
| `.list.max()` | Maximum element | `ListOp[Any]` |
