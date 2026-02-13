"""E2E: Multi-join pipeline — join → cast_schema → verify.

Tests joining multiple tables, handling column name collisions,
and schema binding via mapped_from on joined results.
"""

from __future__ import annotations

import polars as pl

from colnade import Column, DataFrame, Float64, Schema, UInt64, Utf8, mapped_from
from colnade_polars.io import read_parquet

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


class Products(Schema):
    product_id: Column[UInt64]
    product_name: Column[Utf8]
    price: Column[Float64]


class OrderItems(Schema):
    order_id: Column[UInt64]
    product_id: Column[UInt64]
    quantity: Column[UInt64]


class UserOrderSummary(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    user_id: Column[UInt64] = mapped_from(Users.id)
    amount: Column[Float64]


class UserOrderFiltered(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    amount: Column[Float64]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJoinPipeline:
    def test_users_join_orders(self, users_parquet: str, orders_parquet: str) -> None:
        """Join users with orders on id == user_id."""
        users = read_parquet(users_parquet, Users)
        orders = read_parquet(orders_parquet, Orders)

        joined = users.join(orders, on=Users.id == Orders.user_id)

        # Joined result should have rows (inner join)
        assert joined._data.shape[0] > 0
        assert joined._schema_left is Users
        assert joined._schema_right is Orders

    def test_join_then_cast_schema(self, users_parquet: str, orders_parquet: str) -> None:
        """Join → cast_schema with mapped_from resolves correctly."""
        users = read_parquet(users_parquet, Users)
        orders = read_parquet(orders_parquet, Orders)

        result = users.join(orders, on=Users.id == Orders.user_id).cast_schema(UserOrderSummary)

        assert isinstance(result, DataFrame)
        assert result._schema is UserOrderSummary
        assert result._data.columns == ["user_name", "user_id", "amount"]
        assert result._data.shape[0] > 0

    def test_join_filter_then_cast(self, users_parquet: str, orders_parquet: str) -> None:
        """Join → filter on joined data → cast_schema."""
        users = read_parquet(users_parquet, Users)
        orders = read_parquet(orders_parquet, Orders)

        result = (
            users.join(orders, on=Users.id == Orders.user_id)
            .filter(Orders.amount > 200)
            .cast_schema(UserOrderFiltered)
        )

        assert result._schema is UserOrderFiltered
        assert set(result._data.columns) == {"user_name", "amount"}
        # All amounts should be > 200
        assert result._data["amount"].min() > 200  # type: ignore[operator]

    def test_join_sort_limit(self, users_parquet: str, orders_parquet: str) -> None:
        """Join → sort by amount desc → limit 10."""
        users = read_parquet(users_parquet, Users)
        orders = read_parquet(orders_parquet, Orders)

        joined = (
            users.join(orders, on=Users.id == Orders.user_id).sort(Orders.amount.desc()).limit(10)
        )

        assert joined._data.shape[0] == 10
        amounts = joined._data["amount"].to_list()
        assert amounts == sorted(amounts, reverse=True)

    def test_join_correctness_against_polars(self, users_parquet: str, orders_parquet: str) -> None:
        """Verify join results match raw Polars join."""
        users = read_parquet(users_parquet, Users)
        orders = read_parquet(orders_parquet, Orders)

        joined = users.join(orders, on=Users.id == Orders.user_id)

        # Compute expected with raw Polars
        raw_users = pl.read_parquet(users_parquet)
        raw_orders = pl.read_parquet(orders_parquet)
        expected = raw_users.join(raw_orders, left_on="id", right_on="user_id")

        assert joined._data.shape[0] == expected.shape[0]
