"""Joins: joining DataFrames, JoinedDataFrame, cast_schema with mapped_from.

Demonstrates how Colnade handles typed joins between two schemas.
"""

from __future__ import annotations

import polars as pl

from colnade import Column, DataFrame, Float64, Schema, UInt64, Utf8, mapped_from
from colnade_polars import PolarsBackend

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]


class Orders(Schema):
    id: Column[UInt64]
    user_id: Column[UInt64]
    amount: Column[Float64]


# Output schema uses mapped_from to disambiguate joined columns
class UserOrders(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    user_id: Column[UInt64] = mapped_from(Users.id)
    amount: Column[Float64]


# ---------------------------------------------------------------------------
# Create sample data
# ---------------------------------------------------------------------------

users_data = pl.DataFrame(
    {
        "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
        "name": ["Alice", "Bob", "Charlie"],
        "age": pl.Series([30, 25, 35], dtype=pl.UInt64),
    }
)
users = DataFrame(_data=users_data, _schema=Users, _backend=PolarsBackend())

orders_data = pl.DataFrame(
    {
        "id": pl.Series([101, 102, 103, 104], dtype=pl.UInt64),
        "user_id": pl.Series([1, 2, 1, 3], dtype=pl.UInt64),
        "amount": pl.Series([100.0, 200.0, 150.0, 300.0], dtype=pl.Float64),
    }
)
orders = DataFrame(_data=orders_data, _schema=Orders, _backend=PolarsBackend())

print("Users:")
print(users._data)
print()
print("Orders:")
print(orders._data)
print()

# ---------------------------------------------------------------------------
# Join — cross-schema == creates a JoinCondition
# ---------------------------------------------------------------------------

# Users.id == Orders.user_id produces a JoinCondition (not BinOp)
# because the columns belong to different schemas
joined = users.join(orders, on=Users.id == Orders.user_id)  # type: ignore[invalid-argument-type]
print(f"Join result type: {joined!r}")
print(f"Rows after join: {joined._data.shape[0]}")
print(joined._data)
print()

# ---------------------------------------------------------------------------
# cast_schema — flatten JoinedDataFrame to a single schema
# ---------------------------------------------------------------------------

# mapped_from resolves ambiguous columns (both schemas have "id")
# Users.name → "user_name", Users.id → "user_id", amount matches by name
result = joined.cast_schema(UserOrders)
print(f"After cast_schema: {result!r}")
print(f"Columns: {result._data.columns}")
print(result._data)
print()

# ---------------------------------------------------------------------------
# Filter on joined data, then cast
# ---------------------------------------------------------------------------

big_orders = joined.filter(Orders.amount >= 150).cast_schema(UserOrders)
print("Big orders (amount >= 150):")
print(big_orders._data)

print("\nDone!")
