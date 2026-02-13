"""E2E: Null handling pipeline — fill_null, drop_nulls, is_null, assert_non_null.

Tests null handling through the full stack: expression building, translation,
execution, and schema interaction.
"""

from __future__ import annotations

from colnade import Column, Float64, Schema, UInt64, Utf8
from colnade_polars.io import read_parquet

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class NullableUsers(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


class UsersClean(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFillNull:
    def test_fill_null_score(self, nullable_users_parquet: str) -> None:
        """Fill null scores with 0.0."""
        df = read_parquet(nullable_users_parquet, NullableUsers)
        null_count_before = df._data["score"].null_count()

        result = df.with_columns(NullableUsers.score.fill_null(0.0).alias(NullableUsers.score))

        null_count_after = result._data["score"].null_count()
        assert null_count_before > 0
        assert null_count_after == 0

    def test_fill_null_age(self, nullable_users_parquet: str) -> None:
        """Fill null ages with a default value."""
        df = read_parquet(nullable_users_parquet, NullableUsers)
        null_count_before = df._data["age"].null_count()

        result = df.with_columns(NullableUsers.age.fill_null(0).alias(NullableUsers.age))

        null_count_after = result._data["age"].null_count()
        assert null_count_before > 0
        assert null_count_after == 0

    def test_fill_null_then_filter(self, nullable_users_parquet: str) -> None:
        """Fill nulls then filter — no nulls should remain in filtered result."""
        df = read_parquet(nullable_users_parquet, NullableUsers)
        result = df.with_columns(
            NullableUsers.score.fill_null(0.0).alias(NullableUsers.score)
        ).filter(NullableUsers.score > 50)

        assert result._data["score"].null_count() == 0
        assert all(s > 50 for s in result._data["score"].to_list())


class TestDropNulls:
    def test_drop_nulls_single_column(self, nullable_users_parquet: str) -> None:
        """Drop rows with null scores."""
        df = read_parquet(nullable_users_parquet, NullableUsers)
        original_rows = df._data.shape[0]

        result = df.drop_nulls(NullableUsers.score)

        assert result._data.shape[0] < original_rows
        assert result._data["score"].null_count() == 0

    def test_drop_nulls_multiple_columns(self, nullable_users_parquet: str) -> None:
        """Drop rows with nulls in any of the specified columns."""
        df = read_parquet(nullable_users_parquet, NullableUsers)
        result = df.drop_nulls(NullableUsers.age, NullableUsers.score)

        assert result._data["age"].null_count() == 0
        assert result._data["score"].null_count() == 0


class TestIsNull:
    def test_filter_null_rows(self, nullable_users_parquet: str) -> None:
        """Filter to only rows where score is null."""
        df = read_parquet(nullable_users_parquet, NullableUsers)
        result = df.filter(NullableUsers.score.is_null())

        assert result._data.shape[0] > 0
        assert result._data["score"].null_count() == result._data.shape[0]

    def test_filter_not_null_rows(self, nullable_users_parquet: str) -> None:
        """Filter to only rows where score is not null."""
        df = read_parquet(nullable_users_parquet, NullableUsers)
        result = df.filter(NullableUsers.score.is_not_null())

        assert result._data["score"].null_count() == 0

    def test_is_null_and_regular_filter(self, nullable_users_parquet: str) -> None:
        """Combine is_not_null with value filter."""
        df = read_parquet(nullable_users_parquet, NullableUsers)
        result = df.filter(NullableUsers.score.is_not_null() & (NullableUsers.score > 50))

        assert result._data["score"].null_count() == 0
        assert all(s > 50 for s in result._data["score"].to_list())


class TestNullPipeline:
    def test_full_null_cleanup_pipeline(self, nullable_users_parquet: str) -> None:
        """Full pipeline: fill nulls → filter → sort → cast_schema."""
        df = read_parquet(nullable_users_parquet, NullableUsers)
        result = (
            df.with_columns(
                NullableUsers.score.fill_null(0.0).alias(NullableUsers.score),
                NullableUsers.age.fill_null(0).alias(NullableUsers.age),
            )
            .filter(NullableUsers.age > 0)
            .sort(NullableUsers.score.desc())
            .cast_schema(UsersClean)
        )

        assert result._schema is UsersClean
        assert result._data["score"].null_count() == 0
        assert result._data["age"].null_count() == 0
        # ages > 0 means original nulls (filled to 0) are excluded
        assert all(a > 0 for a in result._data["age"].to_list())
