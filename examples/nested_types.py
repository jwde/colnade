"""Nested types: Struct and List column operations.

Demonstrates typed access to struct fields and list operations.
"""

from __future__ import annotations

import colnade as cn
from colnade_polars import from_dict

# ---------------------------------------------------------------------------
# Schemas with nested types
# ---------------------------------------------------------------------------


class Address(cn.Schema):
    city: cn.Column[cn.Utf8]
    zip_code: cn.Column[cn.Utf8]


class UserProfile(cn.Schema):
    id: cn.Column[cn.UInt64]
    name: cn.Column[cn.Utf8]
    address: cn.Column[cn.Struct[Address]]
    tags: cn.Column[cn.List[cn.Utf8]]
    scores: cn.Column[cn.List[cn.Float64]]


class ProfileWithCounts(UserProfile):
    """Extended schema with computed columns from list operations."""

    tag_count: cn.Column[cn.UInt32]
    first_tag: cn.Column[cn.Utf8]
    total_score: cn.Column[cn.Float64]


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

# .list.contains() — filter by list contents
python_users = df.filter(UserProfile.tags.list.contains("python"))
print("Users with 'python' tag:")
print(python_users.select(UserProfile.name, UserProfile.tags))
print()

# .list.len(), .list.get(), .list.sum() — compute into separate columns
# Use a child schema (ProfileWithCounts) so output columns have their own types
enriched = df.with_columns(
    UserProfile.tags.list.len().alias(ProfileWithCounts.tag_count),
    UserProfile.tags.list.get(0).alias(ProfileWithCounts.first_tag),
    UserProfile.scores.list.sum().alias(ProfileWithCounts.total_score),
).cast_schema(ProfileWithCounts)

print("Tag count per user:")
print(enriched.select(UserProfile.name, ProfileWithCounts.tag_count))
print()

print("First tag per user:")
print(enriched.select(UserProfile.name, ProfileWithCounts.first_tag))
print()

print("Total score per user:")
print(enriched.select(UserProfile.name, ProfileWithCounts.total_score))

print("\nDone!")
