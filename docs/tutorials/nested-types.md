# Nested Types

This tutorial demonstrates working with struct and list columns.

!!! tip "Runnable example"
    The complete code is in [`examples/nested_types.py`](https://github.com/jwde/colnade/blob/main/examples/nested_types.py).

## Define schemas with nested types

```python
import colnade as cn

class Address(cn.Schema):
    city: cn.Column[cn.Utf8]
    zip_code: cn.Column[cn.Utf8]

class UserProfile(cn.Schema):
    id: cn.Column[cn.UInt64]
    name: cn.Column[cn.Utf8]
    address: cn.Column[cn.Struct[Address]]
    tags: cn.Column[cn.List[cn.Utf8]]
    scores: cn.Column[cn.List[cn.Float64]]
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
# Check if list contains a value
python_users = df.filter(
    UserProfile.tags.list.contains("python")
)

# Compute list aggregations into new columns
class ProfileWithStats(UserProfile):
    tag_count: cn.Column[cn.UInt32]
    first_tag: cn.Column[cn.Utf8]
    total_score: cn.Column[cn.Float64]

enriched = df.with_columns(
    UserProfile.tags.list.len().alias(ProfileWithStats.tag_count),
    UserProfile.tags.list.get(0).alias(ProfileWithStats.first_tag),
    UserProfile.scores.list.sum().alias(ProfileWithStats.total_score),
).cast_schema(ProfileWithStats)
```

## Available list methods

| Method | Description | Return type |
|--------|-------------|-------------|
| `.list.len()` | Number of elements | `ListOp[UInt32]` |
| `.list.get(i)` | Element at index | `ListOp[DType]` |
| `.list.contains(v)` | Contains value? | `ListOp[Bool]` |
| `.list.sum()` | Sum of elements | `ListOp[DType]` |
| `.list.mean()` | Mean of elements | `ListOp[DType]` |
| `.list.min()` | Minimum element | `ListOp[DType]` |
| `.list.max()` | Maximum element | `ListOp[DType]` |
