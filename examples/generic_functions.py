"""Generic functions: schema-polymorphic utility functions.

Demonstrates how to write functions that work with any schema using TypeVar.
The type checker preserves the concrete schema through generic function calls.
"""

from __future__ import annotations

from colnade import Column, DataFrame, Float64, Schema, UInt64, Utf8
from colnade.schema import S
from colnade_polars import from_dict

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    score: Column[Float64]


class Products(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    price: Column[Float64]


# ---------------------------------------------------------------------------
# Generic utility functions — S is a TypeVar bound to Schema
# ---------------------------------------------------------------------------


def first_n(df: DataFrame[S], n: int) -> DataFrame[S]:
    """Return the first n rows. Works with any schema.

    The type checker knows:
        first_n(users_df, 3) -> DataFrame[Users]
        first_n(products_df, 3) -> DataFrame[Products]
    """
    return df.head(n)


def drop_null_rows(df: DataFrame[S]) -> DataFrame[S]:
    """Drop all rows with any null values. Works with any schema."""
    return df.drop_nulls()


def count_rows(df: DataFrame[S]) -> int:
    """Count rows in any typed DataFrame."""
    return df._data.shape[0]


# ---------------------------------------------------------------------------
# Create sample data
# ---------------------------------------------------------------------------

users = from_dict(
    Users,
    {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [85.0, 92.5, 78.0, 95.0, 88.0],
    },
)

products = from_dict(
    Products,
    {
        "id": [1, 2, 3],
        "name": ["Widget", "Gadget", "Doohickey"],
        "price": [9.99, 24.99, 4.99],
    },
)

# ---------------------------------------------------------------------------
# Use generic functions — schema type is preserved
# ---------------------------------------------------------------------------

# first_n(users, 3) returns DataFrame[Users]
top_users = first_n(users, 3)
print(f"first_n(users, 3) -> {top_users!r}")
print(top_users._data)
print()

# first_n(products, 2) returns DataFrame[Products]
top_products = first_n(products, 2)
print(f"first_n(products, 2) -> {top_products!r}")
print(top_products._data)
print()

# count_rows works on both
print(f"Users count: {count_rows(users)}")
print(f"Products count: {count_rows(products)}")
print()

# drop_null_rows preserves schema
clean_users = drop_null_rows(users)
print(f"drop_null_rows(users) -> {clean_users!r}")
print(f"Rows: {clean_users._data.shape[0]}")

print("\nDone!")
