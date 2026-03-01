"""Joins: joining DataFrames, JoinedDataFrame, cast_schema with mapped_from.

Demonstrates how Colnade handles typed joins between two schemas.
"""

from __future__ import annotations

import colnade as cn
from colnade_polars import from_rows

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Users(cn.Schema):
    id: cn.Column[cn.UInt64]
    name: cn.Column[cn.Utf8]
    age: cn.Column[cn.UInt64]


class Orders(cn.Schema):
    id: cn.Column[cn.UInt64]
    user_id: cn.Column[cn.UInt64]
    amount: cn.Column[cn.Float64]


# Output schema uses mapped_from to disambiguate joined columns
class UserOrders(cn.Schema):
    user_name: cn.Column[cn.Utf8] = cn.mapped_from(Users.name)
    user_id: cn.Column[cn.UInt64] = cn.mapped_from(Users.id)
    amount: cn.Column[cn.Float64]


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
