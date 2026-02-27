"""Full pipeline: a complete ETL example.

Demonstrates an end-to-end pipeline: read data, clean nulls, filter, join,
aggregate, cast to output schema, and write results.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from colnade import Column, Float64, Schema, UInt64, Utf8, mapped_from
from colnade_polars import from_dict, read_parquet, write_parquet

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


class Orders(Schema):
    id: Column[UInt64]
    user_id: Column[UInt64]
    amount: Column[Float64]


class UserOrders(Schema):
    """Intermediate schema after joining users with orders."""

    user_name: Column[Utf8] = mapped_from(Users.name)
    user_id: Column[UInt64] = mapped_from(Users.id)
    amount: Column[Float64] = mapped_from(Orders.amount)


class UserRevenue(Schema):
    """Output schema: per-user revenue summary."""

    user_name: Column[Utf8]
    user_id: Column[UInt64]
    total_amount: Column[Float64]


# ---------------------------------------------------------------------------
# Generate sample data and write to Parquet (from_dict for many rows)
# ---------------------------------------------------------------------------

tmp_dir = Path(tempfile.mkdtemp())

users_df = from_dict(
    Users,
    {
        "id": list(range(1, 11)),
        "name": [
            "Alice",
            "Bob",
            "Charlie",
            "Diana",
            "Eve",
            "Frank",
            "Grace",
            "Henry",
            "Iris",
            "Jack",
        ],
        "age": [30, 25, 35, 28, 40, 22, 33, 45, 27, 31],
        "score": [85.0, 92.5, None, 95.0, 88.0, 76.0, None, 91.0, 82.0, 79.0],
    },
)
users_path = str(tmp_dir / "users.parquet")
write_parquet(users_df, users_path)

orders_df = from_dict(
    Orders,
    {
        "id": list(range(1, 16)),
        "user_id": [1, 2, 1, 3, 5, 2, 8, 1, 4, 6, 3, 5, 9, 10, 7],
        "amount": [
            100.0,
            200.0,
            150.0,
            300.0,
            75.0,
            125.0,
            450.0,
            90.0,
            175.0,
            60.0,
            220.0,
            180.0,
            95.0,
            310.0,
            140.0,
        ],
    },
)
orders_path = str(tmp_dir / "orders.parquet")
write_parquet(orders_df, orders_path)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

print("=== ETL Pipeline ===\n")

# Step 1: Read typed data
users = read_parquet(users_path, Users)
orders = read_parquet(orders_path, Orders)
print(f"Step 1: Read {len(users)} users, {len(orders)} orders")

# Step 2: Clean nulls — fill missing scores with 0
users_clean = users.with_columns(Users.score.fill_null(0.0).alias(Users.score))
print("Step 2: Filled null scores")

# Step 3: Filter — only users aged 25+
active_users = users_clean.filter(Users.age >= 25)
print(f"Step 3: Filtered to {len(active_users)} users aged 25+")

# Step 4: Join users with orders
joined = active_users.join(orders, on=Users.id == Orders.user_id)
print("Step 4: Joined users with orders")

# Step 5: Cast joined data to intermediate schema, then aggregate
user_orders = joined.cast_schema(UserOrders)

revenue = (
    user_orders.group_by(UserOrders.user_name, UserOrders.user_id)
    .agg(UserOrders.amount.sum().alias(UserRevenue.total_amount))
    .cast_schema(UserRevenue)
)

# Step 6: Sort by revenue descending
result = revenue.sort(UserRevenue.total_amount, descending=True)
print("Step 5-6: Aggregated and sorted by revenue")
print()

# Step 7: Display results
print("=== User Revenue Report ===")
print(result)
print()

# Step 8: Write output
output_path = str(tmp_dir / "user_revenue.parquet")
write_parquet(result, output_path)
print(f"Step 8: Wrote results to {output_path}")

# Step 9: Verify — read back and validate
restored = read_parquet(output_path, UserRevenue)
print(f"Step 9: Verified — read back {len(restored)} rows")

print("\nDone!")
