"""E2E: Lazy pipeline — scan → transform → collect → verify.

Tests lazy execution through the full stack: scan_parquet → lazy operations →
collect → verify results match eager equivalent.
"""

from __future__ import annotations

from colnade import Column, DataFrame, Float64, LazyFrame, Schema, UInt64, Utf8
from colnade_polars.io import read_parquet, scan_parquet

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
    name: Column[Utf8]
    score: Column[Float64]


class OrderStats(Schema):
    user_id: Column[UInt64]
    total_amount: Column[Float64]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLazyScan:
    def test_scan_returns_lazyframe(self, users_parquet: str) -> None:
        """scan_parquet returns a LazyFrame with correct schema."""
        lf = scan_parquet(users_parquet, Users)
        assert isinstance(lf, LazyFrame)
        assert lf._schema is Users

    def test_scan_filter_collect(self, users_parquet: str) -> None:
        """Lazy: scan → filter → collect."""
        result = scan_parquet(users_parquet, Users).filter(Users.age > 30).collect()

        assert isinstance(result, DataFrame)
        assert result._schema is Users
        # All ages > 30
        assert all(a > 30 for a in result._data["age"].to_list())

    def test_scan_filter_sort_limit_collect(self, users_parquet: str) -> None:
        """Lazy: scan → filter → sort → limit → collect."""
        result = (
            scan_parquet(users_parquet, Users)
            .filter(Users.age >= 25)
            .sort(Users.age.desc())
            .limit(10)
            .collect()
        )

        assert result._data.shape[0] == 10
        ages = result._data["age"].to_list()
        assert ages == sorted(ages, reverse=True)

    def test_scan_select_cast_collect(self, users_parquet: str) -> None:
        """Lazy: scan → select → cast_schema → collect."""
        result = (
            scan_parquet(users_parquet, Users)
            .select(Users.name, Users.score)
            .cast_schema(UserSummary)
            .collect()
        )

        assert isinstance(result, DataFrame)
        assert result._schema is UserSummary
        assert set(result._data.columns) == {"name", "score"}


class TestLazyMatchesEager:
    def test_filter_results_match(self, users_parquet: str) -> None:
        """Lazy filter results should match eager filter results."""
        eager = read_parquet(users_parquet, Users).filter(Users.age > 30)
        lazy = scan_parquet(users_parquet, Users).filter(Users.age > 30).collect()

        assert eager._data.shape == lazy._data.shape
        assert eager._data["name"].sort().to_list() == lazy._data["name"].sort().to_list()

    def test_sort_results_match(self, users_parquet: str) -> None:
        """Lazy sort results should match eager sort results."""
        eager = read_parquet(users_parquet, Users).sort(Users.age)
        lazy = scan_parquet(users_parquet, Users).sort(Users.age).collect()

        assert eager._data["age"].to_list() == lazy._data["age"].to_list()

    def test_with_columns_results_match(self, users_parquet: str) -> None:
        """Lazy with_columns results should match eager."""
        expr = (Users.score * 2).alias(Users.score)
        eager = read_parquet(users_parquet, Users).with_columns(expr)
        lazy = scan_parquet(users_parquet, Users).with_columns(expr).collect()

        assert eager._data["score"].to_list() == lazy._data["score"].to_list()


class TestLazyToEagerConversion:
    def test_eager_to_lazy_to_eager(self, users_parquet: str) -> None:
        """DataFrame → lazy() → filter → collect() roundtrip."""
        df = read_parquet(users_parquet, Users)
        result = df.lazy().filter(Users.age > 30).collect()

        assert isinstance(result, DataFrame)
        assert result._schema is Users
        assert all(a > 30 for a in result._data["age"].to_list())

    def test_lazy_unique_collect(self, users_parquet: str) -> None:
        """Lazy unique → collect preserves schema and deduplicates."""
        result = scan_parquet(users_parquet, Users).unique(Users.age).collect()

        ages = result._data["age"].to_list()
        assert len(ages) == len(set(ages))

    def test_lazy_drop_nulls_collect(self, nullable_users_parquet: str) -> None:
        """Lazy drop_nulls → collect."""
        from colnade import Schema as _Schema

        class NullableUsers(_Schema):
            id: Column[UInt64]
            name: Column[Utf8]
            age: Column[UInt64]
            score: Column[Float64]

        result = (
            scan_parquet(nullable_users_parquet, NullableUsers)
            .drop_nulls(NullableUsers.score)
            .collect()
        )

        assert result._data["score"].null_count() == 0
