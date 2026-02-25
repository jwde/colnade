"""Cross-layer integration tests spanning expression → translation → execution.

Tests the full roundtrip: Colnade expression tree → PolarsBackend translation →
Polars execution → correct results. Also tests schema validation and generic
functions with concrete schemas.
"""

from __future__ import annotations

import polars as pl
import pytest

from colnade import (
    Column,
    DataFrame,
    Float64,
    Schema,
    SchemaError,
    UInt64,
    Utf8,
    mapped_from,
)
from colnade.schema import S
from colnade_polars.adapter import PolarsBackend

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


class UserSummary(Schema):
    id: Column[UInt64]
    name: Column[Utf8]


class RenamedUsers(Schema):
    user_id: Column[UInt64] = mapped_from(Users.id)
    user_name: Column[Utf8] = mapped_from(Users.name)


class UserOrderSummary(Schema):
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
            "score": pl.Series([85.0, 92.5, 78.0, 95.0, 88.0], dtype=pl.Float64),
        }
    )
    return DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())


def _orders_df() -> DataFrame[Orders]:
    data = pl.DataFrame(
        {
            "id": pl.Series([101, 102, 103, 104, 105], dtype=pl.UInt64),
            "user_id": pl.Series([1, 2, 1, 3, 2], dtype=pl.UInt64),
            "amount": pl.Series([100.0, 200.0, 150.0, 300.0, 75.0], dtype=pl.Float64),
        }
    )
    return DataFrame(_data=data, _schema=Orders, _backend=PolarsBackend())


# ---------------------------------------------------------------------------
# Expression → translation → execution roundtrip
# ---------------------------------------------------------------------------


class TestExprRoundtrip:
    def test_comparison_roundtrip(self) -> None:
        """Column > literal → BinOp → Polars filter → correct rows."""
        df = _users_df()
        result = df.filter(Users.age > 30)
        assert set(result._data["name"].to_list()) == {"Charlie", "Eve"}

    def test_arithmetic_roundtrip(self) -> None:
        """Column * literal → BinOp → Polars with_columns → correct values."""
        df = _users_df()
        result = df.with_columns((Users.age * 2).alias(Users.age))
        assert result._data["age"].to_list() == [60, 50, 70, 56, 80]

    def test_logical_and_roundtrip(self) -> None:
        """(age > 25) & (score > 85) → chained BinOps → correct filter."""
        df = _users_df()
        result = df.filter((Users.age > 25) & (Users.score > 85))
        names = set(result._data["name"].to_list())
        # Alice: age=30>25 but score=85.0 NOT > 85 (strict); Diana: 28>25, 95>85; Eve: 40>25, 88>85
        assert names == {"Diana", "Eve"}

    def test_string_method_roundtrip(self) -> None:
        """str_contains → FunctionCall → Polars str.contains → correct filter."""
        df = _users_df()
        result = df.filter(Users.name.str_starts_with("A"))
        assert result._data["name"].to_list() == ["Alice"]

    def test_aggregation_roundtrip(self) -> None:
        """Column.sum() → Agg → Polars sum → correct result."""
        df = _users_df()
        result = df.group_by(Users.name).agg(Users.score.sum().alias(Users.score))
        # Each user is unique, so sum == original score
        result_sorted = result._data.sort("name")
        assert result_sorted["score"].to_list() == [85.0, 92.5, 78.0, 95.0, 88.0]

    def test_nested_expression_roundtrip(self) -> None:
        """(age + 10) * 2 → nested BinOps → correct computation."""
        df = _users_df()
        result = df.with_columns(((Users.age + 10) * 2).alias(Users.age))
        expected = [(a + 10) * 2 for a in [30, 25, 35, 28, 40]]
        assert result._data["age"].to_list() == expected

    def test_sort_expr_roundtrip(self) -> None:
        """Column.desc() → SortExpr → Polars sort descending → correct order."""
        df = _users_df()
        result = df.sort(Users.age.desc())
        assert result._data["age"].to_list() == [40, 35, 30, 28, 25]


# ---------------------------------------------------------------------------
# Schema validation at read boundary
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_validate_passes_correct_schema(self) -> None:
        """validate() on correctly-typed data succeeds."""
        df = _users_df()
        result = df.validate()
        assert result is df

    def test_validate_catches_missing_column(self) -> None:
        """validate() raises SchemaError for missing columns."""
        data = pl.DataFrame(
            {
                "id": pl.Series([1], dtype=pl.UInt64),
                "name": ["Alice"],
                # Missing: age, score
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        with pytest.raises(SchemaError) as exc_info:
            df.validate()
        assert "age" in exc_info.value.missing_columns
        assert "score" in exc_info.value.missing_columns

    def test_validate_catches_type_mismatch(self) -> None:
        """validate() raises SchemaError for wrong column types."""
        data = pl.DataFrame(
            {
                "id": [1, 2],  # Int64 instead of UInt64
                "name": ["Alice", "Bob"],
                "age": [30, 25],  # Int64 instead of UInt64
                "score": [85.0, 92.5],
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        with pytest.raises(SchemaError) as exc_info:
            df.validate()
        assert len(exc_info.value.type_mismatches) > 0


# ---------------------------------------------------------------------------
# cast_schema with mapped_from on JoinedDataFrame
# ---------------------------------------------------------------------------


class TestCastSchemaOnJoined:
    def test_joined_cast_schema_resolves_mapped_from(self) -> None:
        """cast_schema on JoinedDataFrame resolves mapped_from correctly."""
        users = _users_df()
        orders = _orders_df()

        joined = users.join(orders, on=Users.id == Orders.user_id)
        result = joined.cast_schema(UserOrderSummary)

        assert result._schema is UserOrderSummary
        assert result._data.columns == ["user_name", "user_id", "amount"]
        # Alice has 2 orders (user_id=1), Bob has 2 (user_id=2), Charlie has 1 (user_id=3)
        assert result._data.shape[0] == 5

    def test_joined_cast_schema_data_correct(self) -> None:
        """Verify actual data values after join + cast_schema."""
        users = _users_df()
        orders = _orders_df()

        result = users.join(orders, on=Users.id == Orders.user_id).cast_schema(UserOrderSummary)

        sorted_result = result._data.sort("user_name", "amount")
        alice_orders = sorted_result.filter(pl.col("user_name") == "Alice")
        assert alice_orders["amount"].sort().to_list() == [100.0, 150.0]


# ---------------------------------------------------------------------------
# Generic function with concrete schema
# ---------------------------------------------------------------------------


class TestGenericFunction:
    def test_passthrough_preserves_data(self) -> None:
        """Generic passthrough function (S → S) preserves schema and data."""

        def drop_null_rows(df: DataFrame[S]) -> DataFrame[S]:
            return df.drop_nulls()

        df = _users_df()
        result = drop_null_rows(df)
        assert result._schema is Users
        assert result._data.shape == df._data.shape

    def test_passthrough_with_filter(self) -> None:
        """Generic function that filters preserves schema."""

        def filter_high_scorers(df: DataFrame[S]) -> DataFrame[S]:
            return df.head(3)

        df = _users_df()
        result = filter_high_scorers(df)
        assert result._schema is Users
        assert result._data.shape[0] == 3


# ---------------------------------------------------------------------------
# cast_schema
# ---------------------------------------------------------------------------


class TestCastSchema:
    def test_cast_schema_with_name_match(self) -> None:
        """cast_schema resolves by name when no mapped_from is set."""
        df = _users_df()
        result = df.cast_schema(UserSummary)
        assert result._data.columns == ["id", "name"]
        assert result._data.shape == (5, 2)

    def test_cast_schema_with_mapped_from(self) -> None:
        """cast_schema resolves via mapped_from when set."""
        df = _users_df()
        result = df.cast_schema(RenamedUsers)
        assert result._data.columns == ["user_id", "user_name"]
        assert result._data["user_id"].to_list() == [1, 2, 3, 4, 5]

    def test_cast_schema_child_schema_after_with_columns(self) -> None:
        """with_columns adds a column, cast_schema to child schema picks it up."""

        class EnrichedUsers(Users):
            risk_score: Column[Float64]

        df = _users_df()
        result = df.with_columns(
            (Users.age * 0.1 + Users.score * 0.9).alias(EnrichedUsers.risk_score)
        ).cast_schema(EnrichedUsers)
        assert result._schema is EnrichedUsers
        assert "risk_score" in result._data.columns
        assert result._data.shape == (5, 5)
