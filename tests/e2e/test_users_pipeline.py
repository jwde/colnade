"""E2E: Users pipeline — read → filter → select → cast_schema → verify.

Tests the core workflow of reading typed data, filtering, projecting columns,
and binding to an output schema.
"""

from __future__ import annotations

from pathlib import Path

from colnade import Column, DataFrame, Float64, Schema, UInt64, Utf8, mapped_from
from colnade_polars.io import read_parquet, write_parquet

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


class UserSummary(Schema):
    name: Column[Utf8]
    score: Column[Float64]


class RenamedSummary(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    user_score: Column[Float64] = mapped_from(Users.score)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFilterSelectCast:
    def test_filter_then_select_then_cast(self, users_parquet: str) -> None:
        """Full pipeline: read → filter age > 30 → select name, score → cast_schema."""
        df = read_parquet(users_parquet, Users)
        result = df.filter(Users.age > 30).select(Users.name, Users.score).cast_schema(UserSummary)

        assert isinstance(result, DataFrame)
        assert result._schema is UserSummary
        # All remaining rows have age > 30
        assert result._data.shape[1] == 2
        assert set(result._data.columns) == {"name", "score"}
        assert result._data.shape[0] > 0

    def test_filter_then_sort_then_limit(self, users_parquet: str) -> None:
        """Pipeline: read → filter → sort by score desc → limit 10."""
        df = read_parquet(users_parquet, Users)
        result = df.filter(Users.age >= 25).sort(Users.score.desc()).limit(10)

        assert isinstance(result, DataFrame)
        assert result._schema is Users
        assert result._data.shape[0] == 10
        scores = result._data["score"].to_list()
        # Scores should be in descending order (nulls at end in Polars)
        non_null = [s for s in scores if s is not None]
        assert non_null == sorted(non_null, reverse=True)

    def test_with_columns_then_filter(self, users_parquet: str) -> None:
        """Pipeline: read → compute doubled score → filter by new value."""
        df = read_parquet(users_parquet, Users)
        result = df.with_columns((Users.score * 2).alias(Users.score)).filter(Users.score > 100)

        assert isinstance(result, DataFrame)
        assert result._schema is Users
        # All scores should be > 100 (were doubled)
        scores = result._data["score"].drop_nulls().to_list()
        assert all(s > 100 for s in scores)

    def test_cast_schema_with_mapped_from(self, users_parquet: str) -> None:
        """Pipeline: read → cast_schema using mapped_from renames directly."""
        df = read_parquet(users_parquet, Users)
        result = df.cast_schema(RenamedSummary)

        assert result._schema is RenamedSummary
        assert result._data.columns == ["user_name", "user_score"]
        assert result._data.shape[0] == 100


class TestRoundtrip:
    def test_transform_write_read_back(self, users_parquet: str, tmp_path: Path) -> None:
        """Pipeline: read → filter → cast_schema → write → read back → verify."""
        df = read_parquet(users_parquet, Users)
        summary = df.filter(Users.age > 40).select(Users.name, Users.score).cast_schema(UserSummary)

        out_path = str(tmp_path / "summary.parquet")
        write_parquet(summary, out_path)

        restored = read_parquet(out_path, UserSummary)
        assert restored._schema is UserSummary
        assert restored._data.shape == summary._data.shape
        assert restored._data["name"].to_list() == summary._data["name"].to_list()

    def test_chained_operations_preserve_data(self, users_parquet: str) -> None:
        """Verify data correctness through a chain: filter → unique → sort → head."""
        df = read_parquet(users_parquet, Users)
        result = df.filter(Users.age >= 30).unique(Users.name).sort(Users.name).head(5)

        assert result._schema is Users
        assert result._data.shape[0] == 5
        names = result._data["name"].to_list()
        assert names == sorted(names)
