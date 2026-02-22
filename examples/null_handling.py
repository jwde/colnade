"""Null handling: fill_null, drop_nulls, is_null, is_not_null.

Demonstrates how Colnade handles nullable data with type safety.
"""

from __future__ import annotations

from colnade import Column, Float64, Schema, UInt64, Utf8
from colnade_polars import from_dict

# ---------------------------------------------------------------------------
# Schema with nullable columns
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


# ---------------------------------------------------------------------------
# Create data with nulls (from_dict is natural for nullable columnar data)
# ---------------------------------------------------------------------------

df = from_dict(
    Users,
    {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [30, None, 35, None, 40],
        "score": [85.0, 92.5, None, 95.0, None],
    },
)
print("Original data:")
print(df._data)
print(f"Age nulls: {df._data['age'].null_count()}")
print(f"Score nulls: {df._data['score'].null_count()}")
print()

# ---------------------------------------------------------------------------
# fill_null — replace nulls with a default value
# ---------------------------------------------------------------------------

filled = df.with_columns(
    Users.age.fill_null(0).alias(Users.age),
    Users.score.fill_null(0.0).alias(Users.score),
)
print("After fill_null(0):")
print(filled._data)
print(f"Age nulls: {filled._data['age'].null_count()}")
print(f"Score nulls: {filled._data['score'].null_count()}")
print()

# ---------------------------------------------------------------------------
# drop_nulls — remove rows with nulls in specified columns
# ---------------------------------------------------------------------------

no_null_scores = df.drop_nulls(Users.score)
print("After drop_nulls(Users.score):")
print(no_null_scores._data)
print()

no_nulls_at_all = df.drop_nulls(Users.age, Users.score)
print("After drop_nulls(Users.age, Users.score):")
print(no_nulls_at_all._data)
print()

# ---------------------------------------------------------------------------
# is_null / is_not_null — filter by null status
# ---------------------------------------------------------------------------

null_scores = df.filter(Users.score.is_null())
print("Rows with null scores:")
print(null_scores._data)
print()

valid_scores = df.filter(Users.score.is_not_null())
print("Rows with non-null scores:")
print(valid_scores._data)
print()

# ---------------------------------------------------------------------------
# Combined: fill nulls then filter
# ---------------------------------------------------------------------------

result = df.with_columns(
    Users.score.fill_null(0.0).alias(Users.score),
).filter(Users.score > 50)

print("After fill_null(0.0) then filter(score > 50):")
print(result._data)

print("\nDone!")
