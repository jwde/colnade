"""Negative integration tests — silent correctness and error detail guards.

These tests are CRITICAL for the system. They verify:
1. Operations don't silently produce wrong results (data integrity invariants)
2. Errors contain precise, actionable information (not just "something failed")
3. Edge cases at operation boundaries don't corrupt data

Philosophy: positive tests check that correct things happen; negative tests
check that incorrect things DON'T happen. A filter that returns the right rows
could still be broken if it also includes wrong rows.
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
    name: Column[Utf8]
    score: Column[Float64]


class RenamedUsers(Schema):
    user_id: Column[UInt64] = mapped_from(Users.id)
    user_name: Column[Utf8] = mapped_from(Users.name)


class StrictMatch(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


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


def _duplicated_df() -> DataFrame[Users]:
    data = pl.DataFrame(
        {
            "id": pl.Series([1, 1, 2, 2, 3], dtype=pl.UInt64),
            "name": ["Alice", "Alice", "Bob", "Bob", "Charlie"],
            "age": pl.Series([30, 30, 25, 25, 35], dtype=pl.UInt64),
            "score": pl.Series([85.0, 85.0, 92.5, 92.5, 78.0], dtype=pl.Float64),
        }
    )
    return DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())


# ===================================================================
# 1. POST-FILTER NEGATION GUARDS
#
# After filter(predicate), verify NO remaining row violates the predicate.
# This catches subtle bugs where filter includes extra rows.
# ===================================================================


class TestFilterNegation:
    def test_no_excluded_rows_after_gt(self) -> None:
        """After filter(age > 30), NO row should have age <= 30."""
        df = _users_df()
        result = df.filter(Users.age > 30)
        for age in result._data["age"].to_list():
            assert age > 30, f"Row with age={age} survived filter(age > 30)"

    def test_no_excluded_rows_after_le(self) -> None:
        """After filter(age <= 28), NO row should have age > 28."""
        df = _users_df()
        result = df.filter(Users.age <= 28)
        for age in result._data["age"].to_list():
            assert age <= 28, f"Row with age={age} survived filter(age <= 28)"

    def test_no_excluded_rows_after_eq(self) -> None:
        """After filter(name == 'Alice'), NO other name should appear."""
        df = _users_df()
        result = df.filter(Users.name == "Alice")
        for name in result._data["name"].to_list():
            assert name == "Alice", f"Row with name={name!r} survived filter(name == 'Alice')"

    def test_no_excluded_rows_after_ne(self) -> None:
        """After filter(name != 'Alice'), 'Alice' should NOT appear."""
        df = _users_df()
        result = df.filter(Users.name != "Alice")
        assert "Alice" not in result._data["name"].to_list()

    def test_no_excluded_rows_after_compound(self) -> None:
        """After filter((age >= 28) & (score < 90)), verify BOTH conditions hold."""
        df = _users_df()
        result = df.filter((Users.age >= 28) & (Users.score < 90))
        for row in result._data.iter_rows(named=True):
            assert row["age"] >= 28, f"age={row['age']} violates age >= 28"
            assert row["score"] < 90, f"score={row['score']} violates score < 90"

    def test_filter_preserves_all_qualifying_rows(self) -> None:
        """Filter doesn't DROP rows that should qualify."""
        df = _users_df()
        result = df.filter(Users.age > 30)
        # Charlie (35) and Eve (40) should both be present
        names = set(result._data["name"].to_list())
        assert "Charlie" in names, "Charlie (age=35) missing from filter(age > 30)"
        assert "Eve" in names, "Eve (age=40) missing from filter(age > 30)"
        # And ONLY those two
        assert names == {"Charlie", "Eve"}

    def test_filter_impossible_returns_empty(self) -> None:
        """Filter that no row satisfies returns 0 rows, not an error."""
        df = _users_df()
        result = df.filter(Users.age > 1000)
        assert result._data.shape[0] == 0

    def test_filter_tautology_returns_all(self) -> None:
        """Filter that all rows satisfy returns all rows."""
        df = _users_df()
        result = df.filter(Users.age > 0)
        assert result._data.shape[0] == 5


# ===================================================================
# 2. ROW/COLUMN PRESERVATION INVARIANTS
#
# Operations that shouldn't add/remove rows or columns must not.
# ===================================================================


class TestPreservationInvariants:
    def test_sort_preserves_row_count(self) -> None:
        """Sort must not add or remove rows."""
        df = _users_df()
        result = df.sort(Users.age)
        assert result._data.shape[0] == df._data.shape[0]

    def test_sort_preserves_column_count(self) -> None:
        """Sort must not add or remove columns."""
        df = _users_df()
        result = df.sort(Users.age)
        assert result._data.columns == df._data.columns

    def test_sort_preserves_values(self) -> None:
        """Sort must not change any values, only their order."""
        df = _users_df()
        result = df.sort(Users.age)
        assert sorted(result._data["name"].to_list()) == sorted(df._data["name"].to_list())
        assert sorted(result._data["age"].to_list()) == sorted(df._data["age"].to_list())
        assert sorted(result._data["id"].to_list()) == sorted(df._data["id"].to_list())

    def test_sort_order_is_correct(self) -> None:
        """Sort produces monotonically non-decreasing values."""
        df = _users_df()
        result = df.sort(Users.age)
        ages = result._data["age"].to_list()
        for i in range(len(ages) - 1):
            assert ages[i] <= ages[i + 1], f"Sort violation at index {i}: {ages[i]} > {ages[i + 1]}"

    def test_filter_preserves_column_count(self) -> None:
        """Filter must not change the number of columns."""
        df = _users_df()
        result = df.filter(Users.age > 30)
        assert result._data.columns == df._data.columns

    def test_with_columns_preserves_row_count(self) -> None:
        """with_columns must not add or remove rows."""
        df = _users_df()
        result = df.with_columns((Users.age * 2).alias(Users.age))
        assert result._data.shape[0] == df._data.shape[0]

    def test_with_columns_preserves_other_columns(self) -> None:
        """with_columns must not change columns it doesn't target."""
        df = _users_df()
        result = df.with_columns((Users.age * 2).alias(Users.age))
        # id, name, score should be unchanged
        assert result._data["id"].to_list() == df._data["id"].to_list()
        assert result._data["name"].to_list() == df._data["name"].to_list()
        assert result._data["score"].to_list() == df._data["score"].to_list()

    def test_with_columns_changes_target(self) -> None:
        """with_columns must actually change the targeted column."""
        df = _users_df()
        result = df.with_columns((Users.age * 2).alias(Users.age))
        assert result._data["age"].to_list() != df._data["age"].to_list()
        expected = [a * 2 for a in df._data["age"].to_list()]
        assert result._data["age"].to_list() == expected

    def test_select_removes_unselected_columns(self) -> None:
        """After select(A, B), columns C, D must NOT be present."""
        df = _users_df()
        result = df.select(Users.id, Users.name)
        assert "age" not in result._data.columns
        assert "score" not in result._data.columns

    def test_select_preserves_row_count(self) -> None:
        """Select must not change the number of rows."""
        df = _users_df()
        result = df.select(Users.id, Users.name)
        assert result._data.shape[0] == df._data.shape[0]


# ===================================================================
# 3. DEDUPLICATION INTEGRITY
#
# After unique(), verify NO duplicates exist in the result.
# ===================================================================


class TestUniqueIntegrity:
    def test_unique_removes_all_duplicates(self) -> None:
        """After unique(name), no name should appear more than once."""
        df = _duplicated_df()
        result = df.unique(Users.name)
        names = result._data["name"].to_list()
        assert len(names) == len(set(names)), f"Duplicates remain after unique(): {names}"

    def test_unique_preserves_all_distinct_values(self) -> None:
        """unique() should preserve at least one row for each distinct value."""
        df = _duplicated_df()
        result = df.unique(Users.name)
        assert set(result._data["name"].to_list()) == {"Alice", "Bob", "Charlie"}

    def test_unique_on_already_unique_data(self) -> None:
        """unique() on data without duplicates returns all rows."""
        df = _users_df()
        result = df.unique(Users.name)
        assert result._data.shape[0] == 5


# ===================================================================
# 4. AGGREGATION CORRECTNESS PER TYPE
#
# Each aggregation type must produce the correct mathematical result.
# ===================================================================


class TestAggregationCorrectness:
    def _agg_df(self) -> DataFrame[Users]:
        """DataFrame with known values for manual verification."""
        data = pl.DataFrame(
            {
                "id": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
                "name": ["A", "A", "B", "B"],
                "age": pl.Series([10, 20, 30, 40], dtype=pl.UInt64),
                "score": pl.Series([1.0, 3.0, 5.0, 7.0], dtype=pl.Float64),
            }
        )
        return DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())

    def test_sum_correctness(self) -> None:
        """sum() produces correct sums per group."""
        df = self._agg_df()
        result = df.group_by(Users.name).agg(Users.age.sum().alias(Users.age))
        data = result._data.sort("name")
        assert data["age"].to_list() == [30, 70]  # A: 10+20=30, B: 30+40=70

    def test_mean_correctness(self) -> None:
        """mean() produces correct means per group."""
        df = self._agg_df()
        result = df.group_by(Users.name).agg(Users.score.mean().alias(Users.score))
        data = result._data.sort("name")
        assert data["score"].to_list() == [2.0, 6.0]  # A: (1+3)/2=2, B: (5+7)/2=6

    def test_min_correctness(self) -> None:
        """min() returns smallest value per group."""
        df = self._agg_df()
        result = df.group_by(Users.name).agg(Users.age.min().alias(Users.age))
        data = result._data.sort("name")
        assert data["age"].to_list() == [10, 30]

    def test_max_correctness(self) -> None:
        """max() returns largest value per group."""
        df = self._agg_df()
        result = df.group_by(Users.name).agg(Users.age.max().alias(Users.age))
        data = result._data.sort("name")
        assert data["age"].to_list() == [20, 40]

    def test_count_correctness(self) -> None:
        """count() returns correct count per group."""
        df = self._agg_df()
        result = df.group_by(Users.name).agg(Users.age.count().alias(Users.age))
        data = result._data.sort("name")
        assert data["age"].to_list() == [2, 2]

    def test_first_correctness(self) -> None:
        """first() returns first value per group."""
        df = self._agg_df()
        result = df.group_by(Users.name).agg(Users.age.first().alias(Users.age))
        data = result._data.sort("name")
        assert data["age"].to_list() == [10, 30]

    def test_last_correctness(self) -> None:
        """last() returns last value per group."""
        df = self._agg_df()
        result = df.group_by(Users.name).agg(Users.age.last().alias(Users.age))
        data = result._data.sort("name")
        assert data["age"].to_list() == [20, 40]


# ===================================================================
# 5. SCHEMA ERROR DETAIL PRECISION
#
# SchemaError must contain exact, actionable information — not just
# "something is wrong" but exactly WHAT is wrong.
# ===================================================================


class TestSchemaErrorDetails:
    def test_validate_missing_lists_all_missing(self) -> None:
        """validate() SchemaError lists ALL missing columns, not just the first."""
        data = pl.DataFrame(
            {
                "id": pl.Series([1], dtype=pl.UInt64),
                # missing: name, age, score
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        with pytest.raises(SchemaError) as exc_info:
            df.validate()
        missing = exc_info.value.missing_columns
        assert "name" in missing
        assert "age" in missing
        assert "score" in missing
        assert len(missing) == 3

    def test_validate_type_mismatch_exact_details(self) -> None:
        """validate() SchemaError includes exact column name and mismatched types."""
        data = pl.DataFrame(
            {
                "id": [1, 2],  # Int64 instead of UInt64
                "name": ["Alice", "Bob"],
                "age": pl.Series([30, 25], dtype=pl.UInt64),
                "score": pl.Series([85.0, 92.5], dtype=pl.Float64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        with pytest.raises(SchemaError) as exc_info:
            df.validate()
        mismatches = exc_info.value.type_mismatches
        assert "id" in mismatches
        expected_type, actual_type = mismatches["id"]
        assert expected_type == "UInt64"
        assert actual_type == "Int64"

    def test_cast_schema_missing_exact_columns(self) -> None:
        """cast_schema SchemaError lists exactly which columns are missing."""

        class NeedsEmailAndPhone(Schema):
            email: Column[Utf8]
            phone: Column[Utf8]

        df = _users_df()
        with pytest.raises(SchemaError) as exc_info:
            df.cast_schema(NeedsEmailAndPhone)
        missing = exc_info.value.missing_columns
        assert set(missing) == {"email", "phone"}

    def test_cast_schema_forbid_exact_extra_columns(self) -> None:
        """cast_schema extra='forbid' SchemaError lists exact extra columns."""

        class JustId(Schema):
            id: Column[UInt64]

        df = _users_df()
        with pytest.raises(SchemaError) as exc_info:
            df.cast_schema(JustId, extra="forbid")
        extra = exc_info.value.extra_columns
        assert set(extra) == {"name", "age", "score"}

    def test_schema_error_message_contains_details(self) -> None:
        """SchemaError string message includes the specifics."""
        data = pl.DataFrame(
            {
                "id": pl.Series([1], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        with pytest.raises(SchemaError) as exc_info:
            df.validate()
        msg = str(exc_info.value)
        assert "Missing columns" in msg
        assert "name" in msg


# ===================================================================
# 6. JOIN EDGE CASES
#
# Joins can silently produce wrong results (too many rows, missing rows,
# wrong values in wrong rows).
# ===================================================================


class TestJoinNegative:
    def test_join_no_matches_returns_empty(self) -> None:
        """Inner join with no matching keys returns 0 rows."""
        users = _users_df()
        # Orders with user_ids that don't exist in users
        data = pl.DataFrame(
            {
                "id": pl.Series([201], dtype=pl.UInt64),
                "user_id": pl.Series([999], dtype=pl.UInt64),
                "amount": pl.Series([100.0], dtype=pl.Float64),
            }
        )
        orders = DataFrame(_data=data, _schema=Orders, _backend=PolarsBackend())
        joined = users.join(orders, on=Users.id == Orders.user_id)
        assert joined._data.shape[0] == 0

    def test_join_row_count_matches_expected(self) -> None:
        """Inner join row count equals sum of matches per key."""
        users = _users_df()
        orders = _orders_df()
        joined = users.join(orders, on=Users.id == Orders.user_id)
        # User 1: 2 orders (ids 101, 103), User 2: 2 orders (ids 102, 105),
        # User 3: 1 order (id 104)
        assert joined._data.shape[0] == 5

    def test_join_does_not_duplicate_unmatched_rows(self) -> None:
        """Inner join excludes users with no orders."""
        users = _users_df()
        orders = _orders_df()
        joined = users.join(orders, on=Users.id == Orders.user_id)
        # Users 4 (Diana) and 5 (Eve) have no orders
        user_names = set(joined._data["name"].to_list())
        assert "Diana" not in user_names
        assert "Eve" not in user_names

    def test_join_preserves_values_correctly(self) -> None:
        """Join associates correct amounts with correct users."""
        users = _users_df()
        orders = _orders_df()
        joined = users.join(orders, on=Users.id == Orders.user_id)
        # Alice (id=1) should have amounts 100.0 and 150.0
        alice_rows = joined._data.filter(pl.col("name") == "Alice")
        assert sorted(alice_rows["amount"].to_list()) == [100.0, 150.0]

    def test_joined_cast_schema_ambiguous_column_uses_mapped_from(self) -> None:
        """When both schemas have 'id', mapped_from resolves disambiguation."""

        class JoinResult(Schema):
            user_name: Column[Utf8] = mapped_from(Users.name)
            user_id: Column[UInt64] = mapped_from(Users.id)

        users = _users_df()
        orders = _orders_df()
        result = users.join(orders, on=Users.id == Orders.user_id).cast_schema(JoinResult)
        assert result._data.columns == ["user_name", "user_id"]
        # user_id should come from Users.id (1-5), not Orders.id (101-105)
        assert max(result._data["user_id"].to_list()) <= 5


# ===================================================================
# 7. CAST_SCHEMA INTEGRITY
#
# cast_schema must not silently lose, reorder, or corrupt data.
# ===================================================================


class TestCastSchemaIntegrity:
    def test_cast_preserves_row_count(self) -> None:
        """cast_schema must not add or remove rows."""
        df = _users_df()
        result = df.cast_schema(UserSummary)
        assert result._data.shape[0] == df._data.shape[0]

    def test_cast_preserves_values_for_name_matched(self) -> None:
        """Name-matched columns preserve exact values."""
        df = _users_df()
        result = df.cast_schema(UserSummary)
        assert result._data["name"].to_list() == df._data["name"].to_list()
        assert result._data["score"].to_list() == df._data["score"].to_list()

    def test_cast_mapped_from_preserves_values(self) -> None:
        """mapped_from renames preserve exact values."""
        df = _users_df()
        result = df.cast_schema(RenamedUsers)
        assert result._data["user_id"].to_list() == df._data["id"].to_list()
        assert result._data["user_name"].to_list() == df._data["name"].to_list()

    def test_cast_forbid_exact_match_succeeds(self) -> None:
        """extra='forbid' on exact column match succeeds."""
        df = _users_df()
        result = df.cast_schema(StrictMatch, extra="forbid")
        assert result._data.shape == df._data.shape

    def test_cast_to_wrong_schema_then_validate_fails(self) -> None:
        """If we bypass cast_schema and set wrong schema, validate catches it."""
        data = pl.DataFrame(
            {
                "id": pl.Series([1], dtype=pl.UInt64),
                "name": ["Alice"],
                "age": pl.Series([30], dtype=pl.UInt64),
                "score": pl.Series([85.0], dtype=pl.Float64),
            }
        )

        class WrongSchema(Schema):
            id: Column[UInt64]
            email: Column[Utf8]

        df = DataFrame(_data=data, _schema=WrongSchema, _backend=PolarsBackend())
        with pytest.raises(SchemaError) as exc_info:
            df.validate()
        assert "email" in exc_info.value.missing_columns


# ===================================================================
# 8. BOUNDARY CONDITIONS
#
# Operations at limits must not corrupt data or crash.
# ===================================================================


class TestBoundaryConditions:
    def test_limit_greater_than_row_count(self) -> None:
        """limit(1000) on 5-row DataFrame returns all 5 rows."""
        df = _users_df()
        result = df.limit(1000)
        assert result._data.shape[0] == 5

    def test_head_greater_than_row_count(self) -> None:
        """head(1000) on 5-row DataFrame returns all 5 rows."""
        df = _users_df()
        result = df.head(1000)
        assert result._data.shape[0] == 5

    def test_tail_greater_than_row_count(self) -> None:
        """tail(1000) on 5-row DataFrame returns all 5 rows."""
        df = _users_df()
        result = df.tail(1000)
        assert result._data.shape[0] == 5

    def test_sample_equal_to_row_count(self) -> None:
        """sample(n) where n equals row count returns all rows."""
        df = _users_df()
        result = df.sample(5)
        assert result._data.shape[0] == 5

    def test_limit_zero_returns_empty(self) -> None:
        """limit(0) returns empty DataFrame with correct schema."""
        df = _users_df()
        result = df.limit(0)
        assert result._data.shape[0] == 0
        assert result._data.columns == df._data.columns

    def test_head_zero_returns_empty(self) -> None:
        """head(0) returns empty DataFrame."""
        df = _users_df()
        result = df.head(0)
        assert result._data.shape[0] == 0

    def test_drop_nulls_on_non_null_data(self) -> None:
        """drop_nulls on data with no nulls returns all rows."""
        df = _users_df()
        result = df.drop_nulls(Users.name)
        assert result._data.shape[0] == df._data.shape[0]


# ===================================================================
# 9. LAZY CORRECTNESS GUARDS
#
# Lazy operations must produce identical results to eager equivalents.
# ===================================================================


class TestLazyCorrectness:
    def test_lazy_filter_identical_to_eager(self) -> None:
        """Lazy filter produces same result as eager filter."""
        df = _users_df()
        eager = df.filter(Users.age > 30)
        lazy = df.lazy().filter(Users.age > 30).collect()
        assert eager._data["name"].sort().to_list() == lazy._data["name"].sort().to_list()

    def test_lazy_sort_identical_to_eager(self) -> None:
        """Lazy sort produces same result as eager sort."""
        df = _users_df()
        eager = df.sort(Users.age)
        lazy = df.lazy().sort(Users.age).collect()
        assert eager._data["age"].to_list() == lazy._data["age"].to_list()
        assert eager._data["name"].to_list() == lazy._data["name"].to_list()

    def test_lazy_with_columns_identical_to_eager(self) -> None:
        """Lazy with_columns produces same result as eager."""
        df = _users_df()
        expr = (Users.age * 2).alias(Users.age)
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert eager._data["age"].to_list() == lazy._data["age"].to_list()

    def test_lazy_chain_identical_to_eager_chain(self) -> None:
        """Complex lazy chain produces same result as eager chain."""
        df = _users_df()
        eager = df.filter(Users.age > 25).sort(Users.score.desc()).limit(3)
        lazy = df.lazy().filter(Users.age > 25).sort(Users.score.desc()).limit(3).collect()
        assert eager._data["name"].to_list() == lazy._data["name"].to_list()
        assert eager._data["score"].to_list() == lazy._data["score"].to_list()
