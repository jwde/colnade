"""Joins: joining DataFrames, JoinedDataFrame, cast_schema with mapped_from.

Demonstrates how Colnade handles typed joins between two schemas.
"""

from __future__ import annotations

from colnade import Column, Float64, Schema, UInt64, Utf8, mapped_from
from colnade_polars import from_rows

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

users = from_rows(
    Users,
    [
        Users.Row(id=1, name="Alice", age=30),
        Users.Row(id=2, name="Bob", age=25),
        Users.Row(id=3, name="Charlie", age=35),
    ],
)

orders = from_rows(
    Orders,
    [
        Orders.Row(id=101, user_id=1, amount=100.0),
        Orders.Row(id=102, user_id=2, amount=200.0),
        Orders.Row(id=103, user_id=1, amount=150.0),
        Orders.Row(id=104, user_id=3, amount=300.0),
    ],
)

print("Users:")
print(users)
print()
print("Orders:")
print(orders)
print()

# ---------------------------------------------------------------------------
# Join — cross-schema == creates a JoinCondition
# ---------------------------------------------------------------------------

# Users.id == Orders.user_id produces a JoinCondition (not BinOp)
# because the columns belong to different schemas
joined = users.join(orders, on=Users.id == Orders.user_id)
print(f"Join result: {joined!r}")
print()

# ---------------------------------------------------------------------------
# cast_schema — flatten JoinedDataFrame to a single schema
# ---------------------------------------------------------------------------

# mapped_from resolves ambiguous columns (both schemas have "id")
# Users.name → "user_name", Users.id → "user_id", amount matches by name
result = joined.cast_schema(UserOrders)
print(f"After cast_schema: {result!r}")
print(result)
print()

# ---------------------------------------------------------------------------
# Filter on joined data, then cast
# ---------------------------------------------------------------------------

big_orders = joined.filter(Orders.amount >= 150).cast_schema(UserOrders)
print("Big orders (amount >= 150):")
print(big_orders)

print("\nDone!")
