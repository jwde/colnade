"""Nested types: Struct and List column operations.

Demonstrates typed access to struct fields and list operations.
"""

from __future__ import annotations

from colnade import Column, Float64, List, Schema, Struct, UInt64, Utf8
from colnade_polars import from_dict

# ---------------------------------------------------------------------------
# Schemas with nested types
# ---------------------------------------------------------------------------


class Address(Schema):
    city: Column[Utf8]
    zip_code: Column[Utf8]


class UserProfile(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    address: Column[Struct[Address]]
    tags: Column[List[Utf8]]
    scores: Column[List[Float64]]


# ---------------------------------------------------------------------------
# Create data with struct and list columns (from_dict is natural for nested types)
# ---------------------------------------------------------------------------

df = from_dict(
    UserProfile,
    {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "address": [
            {"city": "New York", "zip_code": "10001"},
            {"city": "San Francisco", "zip_code": "94102"},
            {"city": "New York", "zip_code": "10002"},
        ],
        "tags": [["python", "data"], ["rust", "systems"], ["python", "ml", "data"]],
        "scores": [[85.0, 90.0], [92.5, 88.0, 95.0], [78.0]],
    },
)
print("User profiles:")
print(df)
print()

# ---------------------------------------------------------------------------
# Struct field access — typed field references
# ---------------------------------------------------------------------------

# Access struct fields using .field() with a Column from the struct's schema
# UserProfile.address.field(Address.city) creates a StructFieldAccess node
new_yorkers = df.filter(UserProfile.address.field(Address.city) == "New York")
print("Users in New York:")
print(new_yorkers)
print()

# ---------------------------------------------------------------------------
# List operations — typed list access via .list property
# ---------------------------------------------------------------------------

# .list.len() — count elements in each list
print("Tag counts:")
tag_lengths = df.with_columns(UserProfile.tags.list.len().alias(UserProfile.tags))
print(tag_lengths.select(UserProfile.name, UserProfile.tags))
print()

# .list.contains() — check if list contains a value
python_users = df.filter(UserProfile.tags.list.contains("python"))
print("Users with 'python' tag:")
print(python_users.select(UserProfile.name, UserProfile.tags))
print()

# .list.get() — get element by index
first_tags = df.with_columns(UserProfile.tags.list.get(0).alias(UserProfile.tags))
print("First tag per user:")
print(first_tags.select(UserProfile.name, UserProfile.tags))
print()

# .list.sum() — sum list elements (for numeric lists)
score_totals = df.with_columns(UserProfile.scores.list.sum().alias(UserProfile.scores))
print("Total scores per user:")
print(score_totals.select(UserProfile.name, UserProfile.scores))

print("\nDone!")
