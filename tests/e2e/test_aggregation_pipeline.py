"""E2E: Aggregation pipeline — group_by → agg → cast_schema → verify.

Tests grouped aggregations with multiple expressions, schema binding via
cast_schema, and verification of aggregation correctness.
"""

from __future__ import annotations

import polars as pl

from colnade import Column, DataFrame, Float64, Schema, UInt32, UInt64, Utf8
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


class AgeGroupStats(Schema):
    age: Column[UInt64]
    avg_score: Column[Float64]


class OrderStats(Schema):
    user_id: Column[UInt64]
    total_amount: Column[Float64]
    order_count: Column[UInt32]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGroupByAgg:
    def test_single_agg(self, users_parquet: str) -> None:
        """Group by age, compute mean score."""
        df = read_parquet(users_parquet, Users)
        result = df.group_by(Users.age).agg(Users.score.mean().alias(Users.score))

        assert isinstance(result, DataFrame)
        # Every age group should have a mean score
        assert result._data.shape[0] > 0
        assert "age" in result._data.columns
        assert "score" in result._data.columns

    def test_agg_then_cast_schema(self, users_parquet: str) -> None:
        """Group by age → agg mean score → cast_schema to AgeGroupStats."""
        df = read_parquet(users_parquet, Users)
        result = (
            df.group_by(Users.age)
            .agg(Users.score.mean().as_column(AgeGroupStats.avg_score))
            .cast_schema(AgeGroupStats)
        )

        assert result._schema is AgeGroupStats
        assert set(result._data.columns) == {"age", "avg_score"}
        assert result._data.shape[0] > 0

    def test_multi_agg(self, orders_parquet: str) -> None:
        """Group orders by user_id, compute sum and count."""
        df = read_parquet(orders_parquet, Orders)
        result = (
            df.group_by(Orders.user_id)
            .agg(
                Orders.amount.sum().as_column(OrderStats.total_amount),
                Orders.amount.count().as_column(OrderStats.order_count),
            )
            .cast_schema(OrderStats)
        )

        assert result._schema is OrderStats
        assert set(result._data.columns) == {"user_id", "total_amount", "order_count"}
        # Every user_id in orders should appear
        assert result._data.shape[0] > 0
        # Total amounts should all be positive
        assert result._data["total_amount"].min() > 0  # type: ignore[operator]

    def test_filter_then_agg(self, users_parquet: str) -> None:
        """Filter users first, then aggregate — verify filter applied before agg."""
        df = read_parquet(users_parquet, Users)
        all_result = df.group_by(Users.age).agg(Users.id.count().alias(Users.id))
        filtered_result = (
            df.filter(Users.score > 50).group_by(Users.age).agg(Users.id.count().alias(Users.id))
        )

        # Filtered result should have fewer or equal total rows
        total_all = all_result._data["id"].sum()
        total_filtered = filtered_result._data["id"].sum()
        assert total_filtered <= total_all  # type: ignore[operator]

    def test_agg_correctness_manual(self, orders_parquet: str) -> None:
        """Verify aggregation results against manual Polars computation."""
        df = read_parquet(orders_parquet, Orders)
        result = df.group_by(Orders.user_id).agg(
            Orders.amount.sum().as_column(OrderStats.total_amount)
        )

        # Compute expected with raw Polars
        raw = pl.read_parquet(orders_parquet)
        expected = raw.group_by("user_id").agg(pl.col("amount").sum().alias("total_amount"))

        result_sorted = result._data.sort("user_id")
        expected_sorted = expected.sort("user_id")
        assert result_sorted["user_id"].to_list() == expected_sorted["user_id"].to_list()
        assert result_sorted["total_amount"].to_list() == expected_sorted["total_amount"].to_list()
