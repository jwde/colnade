"""Generic functions: schema-polymorphic utility functions.

Demonstrates how to write functions that work with any schema using TypeVar.
The type checker preserves the concrete schema through generic function calls.
"""

from __future__ import annotations

from colnade import Column, DataFrame, Float64, Schema, UInt64, Utf8
from colnade.schema import S
from colnade_polars import from_rows

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
    return len(df)


# ---------------------------------------------------------------------------
# Create sample data
# ---------------------------------------------------------------------------

users = from_rows(
    Users,
    [
        Users.Row(id=1, name="Alice", score=85.0),
        Users.Row(id=2, name="Bob", score=92.5),
        Users.Row(id=3, name="Charlie", score=78.0),
        Users.Row(id=4, name="Diana", score=95.0),
        Users.Row(id=5, name="Eve", score=88.0),
    ],
)

products = from_rows(
    Products,
    [
        Products.Row(id=1, name="Widget", price=9.99),
        Products.Row(id=2, name="Gadget", price=24.99),
        Products.Row(id=3, name="Doohickey", price=4.99),
    ],
)

# ---------------------------------------------------------------------------
# Use generic functions — schema type is preserved
# ---------------------------------------------------------------------------

# first_n(users, 3) returns DataFrame[Users]
top_users = first_n(users, 3)
print(f"first_n(users, 3) -> {top_users!r}")
print(top_users)
print()

# first_n(products, 2) returns DataFrame[Products]
top_products = first_n(products, 2)
print(f"first_n(products, 2) -> {top_products!r}")
print(top_products)
print()

# count_rows works on both
print(f"Users count: {count_rows(users)}")
print(f"Products count: {count_rows(products)}")
print()

# drop_null_rows preserves schema
clean_users = drop_null_rows(users)
print(f"drop_null_rows(users) -> {clean_users!r}")
print(f"Rows: {len(clean_users)}")

print("\nDone!")
