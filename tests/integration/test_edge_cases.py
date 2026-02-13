"""Edge case integration tests spanning multiple components.

Tests corner cases: column name collisions in joins, empty DataFrames,
single-row aggregations, many-column schemas, long expression chains,
cast_schema extra="forbid"/"drop".
"""

from __future__ import annotations

import polars as pl
import pytest

from colnade import Column, DataFrame, Float64, Schema, SchemaError, UInt64, Utf8, mapped_from
from colnade_polars.adapter import PolarsBackend

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


class UserSummary(Schema):
    id: Column[UInt64]
    name: Column[Utf8]


class StrictSummary(Schema):
    id: Column[UInt64]
    name: Column[Utf8]


class JoinOutput(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    user_id: Column[UInt64] = mapped_from(Users.id)
    amount: Column[Float64]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _users_df() -> DataFrame[Users]:
    data = pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3, 4, 5], dtype=pl.UInt64),
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": pl.Series([30, 25, 35, 28, 40], dtype=pl.UInt64),
        }
    )
    return DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())


def _orders_df() -> DataFrame[Orders]:
    data = pl.DataFrame(
        {
            "id": pl.Series([101, 102, 103], dtype=pl.UInt64),
            "user_id": pl.Series([1, 2, 1], dtype=pl.UInt64),
            "amount": pl.Series([100.0, 200.0, 150.0], dtype=pl.Float64),
        }
    )
    return DataFrame(_data=data, _schema=Orders, _backend=PolarsBackend())


# ---------------------------------------------------------------------------
# Column name collision in joins
# ---------------------------------------------------------------------------


class TestJoinCollision:
    def test_join_with_shared_column_name(self) -> None:
        """Both Users and Orders have 'id' — join handles this via disambiguation."""
        users = _users_df()
        orders = _orders_df()

        joined = users.join(orders, on=Users.id == Orders.user_id)
        # Polars suffixes duplicate columns — 'id' from both schemas
        assert joined._data.shape[0] == 3  # users 1 and 2 matched
        # The join should complete without error
        assert joined._schema_left is Users
        assert joined._schema_right is Orders

    def test_join_collision_cast_schema_uses_mapped_from(self) -> None:
        """cast_schema on joined data with collision uses mapped_from to disambiguate."""
        users = _users_df()
        orders = _orders_df()

        result = users.join(orders, on=Users.id == Orders.user_id).cast_schema(JoinOutput)

        assert result._data.columns == ["user_name", "user_id", "amount"]
        assert result._data.shape[0] == 3


# ---------------------------------------------------------------------------
# Empty DataFrame
# ---------------------------------------------------------------------------


class TestEmptyDataFrame:
    def test_empty_filter(self) -> None:
        """Filter that produces 0 rows works without error."""
        df = _users_df()
        result = df.filter(Users.age > 100)
        assert result._data.shape[0] == 0
        assert result._schema is Users

    def test_empty_sort(self) -> None:
        """Sort on empty DataFrame works."""
        df = _users_df()
        result = df.filter(Users.age > 100).sort(Users.age)
        assert result._data.shape[0] == 0

    def test_empty_with_columns(self) -> None:
        """with_columns on empty DataFrame works."""
        df = _users_df()
        result = df.filter(Users.age > 100).with_columns((Users.age * 2).alias(Users.age))
        assert result._data.shape[0] == 0
        assert "age" in result._data.columns

    def test_empty_select(self) -> None:
        """select on empty DataFrame works."""
        df = _users_df()
        result = df.filter(Users.age > 100).select(Users.id, Users.name)
        assert result._data.shape[0] == 0
        assert result._data.columns == ["id", "name"]

    def test_empty_unique(self) -> None:
        """unique on empty DataFrame works."""
        df = _users_df()
        result = df.filter(Users.age > 100).unique(Users.name)
        assert result._data.shape[0] == 0

    def test_empty_cast_schema(self) -> None:
        """cast_schema on empty DataFrame works."""
        df = _users_df()
        result = df.filter(Users.age > 100).cast_schema(UserSummary)
        assert result._data.shape[0] == 0
        assert result._schema is UserSummary

    def test_empty_lazy_collect(self) -> None:
        """Empty lazy → collect works."""
        df = _users_df()
        result = df.lazy().filter(Users.age > 100).collect()
        assert result._data.shape[0] == 0


# ---------------------------------------------------------------------------
# Single-row DataFrame
# ---------------------------------------------------------------------------


class TestSingleRow:
    def test_single_row_operations(self) -> None:
        """All operations work on a single-row DataFrame."""
        data = pl.DataFrame(
            {
                "id": pl.Series([1], dtype=pl.UInt64),
                "name": ["Alice"],
                "age": pl.Series([30], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())

        # filter
        assert df.filter(Users.age > 20)._data.shape[0] == 1
        assert df.filter(Users.age > 50)._data.shape[0] == 0

        # sort
        assert df.sort(Users.age)._data.shape[0] == 1

        # with_columns
        result = df.with_columns((Users.age + 1).alias(Users.age))
        assert result._data["age"].to_list() == [31]

        # select
        assert df.select(Users.name)._data.columns == ["name"]

        # unique (single row is already unique)
        assert df.unique(Users.name)._data.shape[0] == 1

    def test_single_row_aggregation(self) -> None:
        """Aggregation on single row returns correct value."""
        data = pl.DataFrame(
            {
                "id": pl.Series([1], dtype=pl.UInt64),
                "name": ["Alice"],
                "age": pl.Series([30], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        result = df.group_by(Users.name).agg(Users.age.sum().alias(Users.age))
        assert result._data["age"].to_list() == [30]


# ---------------------------------------------------------------------------
# Schema with many columns
# ---------------------------------------------------------------------------


class TestManyColumns:
    def test_twenty_column_schema(self) -> None:
        """Schema with 20 columns works correctly."""

        class Wide(Schema):
            c01: Column[UInt64]
            c02: Column[UInt64]
            c03: Column[UInt64]
            c04: Column[UInt64]
            c05: Column[UInt64]
            c06: Column[UInt64]
            c07: Column[UInt64]
            c08: Column[UInt64]
            c09: Column[UInt64]
            c10: Column[UInt64]
            c11: Column[UInt64]
            c12: Column[UInt64]
            c13: Column[UInt64]
            c14: Column[UInt64]
            c15: Column[UInt64]
            c16: Column[UInt64]
            c17: Column[UInt64]
            c18: Column[UInt64]
            c19: Column[UInt64]
            c20: Column[UInt64]

        cols = {f"c{i:02d}": pl.Series([i], dtype=pl.UInt64) for i in range(1, 21)}
        data = pl.DataFrame(cols)
        df = DataFrame(_data=data, _schema=Wide, _backend=PolarsBackend())

        # validate succeeds
        df.validate()

        # filter works
        result = df.filter(Wide.c01 > 0)
        assert result._data.shape == (1, 20)

        # select subset works
        result = df.select(Wide.c01, Wide.c10, Wide.c20)
        assert result._data.columns == ["c01", "c10", "c20"]


# ---------------------------------------------------------------------------
# Long expression chains
# ---------------------------------------------------------------------------


class TestLongExpressionChain:
    def test_deeply_nested_arithmetic(self) -> None:
        """Deeply nested arithmetic expression doesn't stack overflow."""
        df = _users_df()
        # Build: ((((age + 1) + 1) + 1) ... + 1) — 50 levels deep
        expr = Users.age + 0
        for _ in range(50):
            expr = expr + 1
        result = df.with_columns(expr.alias(Users.age))
        # Original ages were [30, 25, 35, 28, 40], each gets +50
        assert result._data["age"].to_list() == [80, 75, 85, 78, 90]

    def test_chained_filter_operations(self) -> None:
        """Multiple chained filter operations work correctly."""
        df = _users_df()
        result = (
            df.filter(Users.age >= 25)
            .filter(Users.age <= 40)
            .filter(Users.name.str_starts_with("A") | Users.name.str_starts_with("E"))
        )
        names = set(result._data["name"].to_list())
        assert names == {"Alice", "Eve"}


# ---------------------------------------------------------------------------
# cast_schema extra="forbid" and extra="drop"
# ---------------------------------------------------------------------------


class TestCastSchemaExtra:
    def test_extra_drop_default(self) -> None:
        """cast_schema with extra='drop' (default) silently drops extra columns."""
        df = _users_df()
        result = df.cast_schema(UserSummary)

        assert result._data.columns == ["id", "name"]
        assert result._data.shape == (5, 2)

    def test_extra_forbid_raises_on_extra_columns(self) -> None:
        """cast_schema with extra='forbid' raises SchemaError for extra columns."""
        df = _users_df()
        with pytest.raises(SchemaError) as exc_info:
            df.cast_schema(UserSummary, extra="forbid")
        # "age" is an extra column not in UserSummary
        assert "age" in exc_info.value.extra_columns

    def test_extra_forbid_passes_when_exact(self) -> None:
        """cast_schema with extra='forbid' succeeds when columns match exactly."""
        df = _users_df()
        selected = df.select(Users.id, Users.name)
        result = selected.cast_schema(UserSummary, extra="forbid")
        assert result._data.columns == ["id", "name"]

    def test_cast_schema_missing_column_raises(self) -> None:
        """cast_schema raises SchemaError when target has unmapped columns."""

        class NeedsEmail(Schema):
            id: Column[UInt64]
            email: Column[Utf8]

        df = _users_df()
        with pytest.raises(SchemaError) as exc_info:
            df.cast_schema(NeedsEmail)
        assert "email" in exc_info.value.missing_columns

    def test_cast_schema_explicit_mapping(self) -> None:
        """cast_schema with explicit mapping dict overrides name matching."""

        class Output(Schema):
            person_name: Column[Utf8]
            person_id: Column[UInt64]

        df = _users_df()
        result = df.cast_schema(
            Output,
            mapping={Output.person_name: Users.name, Output.person_id: Users.id},
        )
        assert result._data.columns == ["person_name", "person_id"]
        assert result._data["person_name"].to_list() == ["Alice", "Bob", "Charlie", "Diana", "Eve"]
