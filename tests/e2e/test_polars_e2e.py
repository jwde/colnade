"""End-to-end tests for full Polars pipeline execution."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from colnade import Column, DataFrame, Schema, UInt64, Utf8, mapped_from
from colnade_polars.io import read_parquet, write_parquet

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]


class Orders(Schema):
    order_id: Column[UInt64]
    user_id: Column[UInt64]
    amount: Column[UInt64]


class UserSummary(Schema):
    id: Column[UInt64]
    name: Column[Utf8]


class UserOrderSummary(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    total_amount: Column[UInt64]


class UserSpend(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    user_id: Column[UInt64] = mapped_from(Users.id)
    amount: Column[UInt64]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def users_path(tmp_path: Path) -> str:
    path = str(tmp_path / "users.parquet")
    pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3, 4, 5], dtype=pl.UInt64),
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": pl.Series([30, 25, 35, 28, 40], dtype=pl.UInt64),
        }
    ).write_parquet(path)
    return path


@pytest.fixture
def orders_path(tmp_path: Path) -> str:
    path = str(tmp_path / "orders.parquet")
    pl.DataFrame(
        {
            "order_id": pl.Series([101, 102, 103, 104], dtype=pl.UInt64),
            "user_id": pl.Series([1, 2, 1, 3], dtype=pl.UInt64),
            "amount": pl.Series([100, 200, 150, 300], dtype=pl.UInt64),
        }
    ).write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# E2E: read → filter → select → cast_schema → verify values
# ---------------------------------------------------------------------------


class TestFilterSelectPipeline:
    def test_full_pipeline(self, users_path: str) -> None:
        df = read_parquet(users_path, Users)

        result = df.filter(Users.age > 28).select(Users.id, Users.name).cast_schema(UserSummary)

        assert isinstance(result, DataFrame)
        assert result._schema is UserSummary
        assert result._data.shape[0] == 3
        names = set(result._data["name"].to_list())
        assert names == {"Alice", "Charlie", "Eve"}


# ---------------------------------------------------------------------------
# E2E: read → group_by → agg → cast_schema → verify
# ---------------------------------------------------------------------------


class TestGroupByPipeline:
    def test_group_by_agg_pipeline(self, users_path: str, orders_path: str) -> None:
        orders = read_parquet(orders_path, Orders)

        # Group orders by user_id, sum amount
        agg_result = orders.group_by(Orders.user_id).agg(Orders.amount.sum().alias(Orders.amount))

        # Verify aggregation works correctly
        data = agg_result._data.sort("user_id")
        assert data["user_id"].to_list() == [1, 2, 3]
        assert data["amount"].to_list() == [250, 200, 300]


# ---------------------------------------------------------------------------
# E2E: join pipeline — read two files → join → cast_schema → verify
# ---------------------------------------------------------------------------


class TestJoinPipeline:
    def test_join_and_cast(self, users_path: str, orders_path: str) -> None:
        users = read_parquet(users_path, Users)
        orders = read_parquet(orders_path, Orders)

        joined = users.join(orders, on=Users.id == Orders.user_id)
        result = joined.cast_schema(UserSpend)

        assert isinstance(result, DataFrame)
        assert result._schema is UserSpend
        assert result._data.columns == ["user_name", "user_id", "amount"]
        # Alice has 2 orders, Bob 1, Charlie 1 → 4 rows
        assert result._data.shape[0] == 4


# ---------------------------------------------------------------------------
# E2E: roundtrip — read → transform → write → read back → verify
# ---------------------------------------------------------------------------


class TestRoundtripPipeline:
    def test_transform_and_roundtrip(self, users_path: str, tmp_path: Path) -> None:
        df = read_parquet(users_path, Users)
        filtered = df.filter(Users.age >= 30).select(Users.id, Users.name).cast_schema(UserSummary)

        out_path = str(tmp_path / "output.parquet")
        write_parquet(filtered, out_path)

        result = read_parquet(out_path, UserSummary)
        assert result._data.shape[0] == 3
        assert set(result._data["name"].to_list()) == {"Alice", "Charlie", "Eve"}
