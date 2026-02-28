"""Integration tests for Polars execution via the DataFrame layer."""

from __future__ import annotations

import polars as pl
import pytest

from colnade import (
    Column,
    DataFrame,
    LazyFrame,
    Schema,
    SchemaError,
    UInt64,
    Utf8,
    concat,
    lit,
    mapped_from,
    when,
)
from colnade_polars.adapter import PolarsBackend

# ---------------------------------------------------------------------------
# Test schemas
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


class RenamedUsers(Schema):
    user_id: Column[UInt64] = mapped_from(Users.id)
    user_name: Column[Utf8] = mapped_from(Users.name)


class AgeStats(Schema):
    name: Column[Utf8]
    total_age: Column[UInt64]


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
            "order_id": pl.Series([101, 102, 103], dtype=pl.UInt64),
            "user_id": pl.Series([1, 2, 1], dtype=pl.UInt64),
            "amount": pl.Series([100, 200, 150], dtype=pl.UInt64),
        }
    )
    return DataFrame(_data=data, _schema=Orders, _backend=PolarsBackend())


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


class TestFilter:
    def test_filter_by_comparison(self) -> None:
        df = _users_df()
        result = df.filter(Users.age > 30)
        assert isinstance(result, DataFrame)
        assert result._data.shape == (2, 3)
        assert set(result._data["name"].to_list()) == {"Charlie", "Eve"}

    def test_filter_chained(self) -> None:
        df = _users_df()
        result = df.filter((Users.age >= 25) & (Users.age <= 35))
        assert result._data.shape == (4, 3)


# ---------------------------------------------------------------------------
# sort
# ---------------------------------------------------------------------------


class TestSort:
    def test_sort_by_column(self) -> None:
        df = _users_df()
        result = df.sort(Users.age)
        names = result._data["name"].to_list()
        assert names == ["Bob", "Diana", "Alice", "Charlie", "Eve"]

    def test_sort_descending(self) -> None:
        df = _users_df()
        result = df.sort(Users.age, descending=True)
        names = result._data["name"].to_list()
        assert names == ["Eve", "Charlie", "Alice", "Diana", "Bob"]

    def test_sort_by_sort_expr(self) -> None:
        df = _users_df()
        result = df.sort(Users.age.desc())
        names = result._data["name"].to_list()
        assert names == ["Eve", "Charlie", "Alice", "Diana", "Bob"]


# ---------------------------------------------------------------------------
# limit / head / tail / sample
# ---------------------------------------------------------------------------


class TestSlicing:
    def test_limit(self) -> None:
        df = _users_df()
        result = df.limit(2)
        assert result._data.shape[0] == 2

    def test_head(self) -> None:
        df = _users_df()
        result = df.head(3)
        assert result._data.shape[0] == 3

    def test_tail(self) -> None:
        df = _users_df()
        result = df.tail(2)
        assert result._data.shape[0] == 2

    def test_sample(self) -> None:
        df = _users_df()
        result = df.sample(3)
        assert result._data.shape[0] == 3


# ---------------------------------------------------------------------------
# unique / drop_nulls
# ---------------------------------------------------------------------------


class TestUniqueDropNulls:
    def test_unique(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([1, 1, 2], dtype=pl.UInt64),
                "name": ["Alice", "Alice", "Bob"],
                "age": pl.Series([30, 30, 25], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        result = df.unique(Users.name)
        assert result._data.shape[0] == 2

    def test_unique_no_columns(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([1, 1, 2], dtype=pl.UInt64),
                "name": ["Alice", "Alice", "Bob"],
                "age": pl.Series([30, 30, 25], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        result = df.unique()
        assert result._data.shape[0] == 2

    def test_drop_nulls(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
                "name": ["Alice", None, "Charlie"],
                "age": pl.Series([30, 25, 35], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        result = df.drop_nulls(Users.name)
        assert result._data.shape[0] == 2


# ---------------------------------------------------------------------------
# with_columns
# ---------------------------------------------------------------------------


class TestWithColumns:
    def test_with_columns_computed(self) -> None:
        df = _users_df()
        result = df.with_columns((Users.age * 2).alias(Users.age))
        ages = result._data["age"].to_list()
        assert ages == [60, 50, 70, 56, 80]


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


class TestSelect:
    def test_select_subset(self) -> None:
        df = _users_df()
        result = df.select(Users.id, Users.name)
        assert result._data.columns == ["id", "name"]
        assert result._data.shape == (5, 2)


# ---------------------------------------------------------------------------
# group_by + agg
# ---------------------------------------------------------------------------


class TestAggFunctions:
    def test_std(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
                "name": ["Alice", "Bob", "Charlie"],
                "age": pl.Series([30, 25, 35], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        result = df.agg(Users.age.std().alias(Users.age))
        assert result._data["age"][0] > 0

    def test_var(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
                "name": ["Alice", "Bob", "Charlie"],
                "age": pl.Series([30, 25, 35], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        result = df.agg(Users.age.var().alias(Users.age))
        assert result._data["age"][0] > 0

    def test_n_unique(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
                "name": ["Alice", "Alice", "Bob"],
                "age": pl.Series([30, 25, 35], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        result = df.agg(Users.name.n_unique().alias(Users.name))
        assert result._data["name"][0] == 2


class TestGroupByAgg:
    def test_group_by_sum(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
                "name": ["Alice", "Alice", "Bob", "Bob"],
                "age": pl.Series([10, 20, 30, 40], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        result = df.group_by(Users.name).agg(Users.age.sum().alias(Users.age))
        result_data = result._data.sort("name")
        assert result_data["name"].to_list() == ["Alice", "Bob"]
        assert result_data["age"].to_list() == [30, 70]


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------


class TestJoin:
    def test_inner_join(self) -> None:
        users = _users_df()
        orders = _orders_df()
        joined = users.join(orders, on=Users.id == Orders.user_id)
        # Users 1 (Alice) has 2 orders, User 2 (Bob) has 1 order
        assert joined._data.shape[0] == 3


# ---------------------------------------------------------------------------
# cast_schema
# ---------------------------------------------------------------------------


class TestCastSchema:
    def test_cast_schema_name_match(self) -> None:
        df = _users_df()
        result = df.cast_schema(UserSummary)
        assert isinstance(result, DataFrame)
        assert result._data.columns == ["id", "name"]
        assert result._data.shape == (5, 2)

    def test_cast_schema_mapped_from(self) -> None:
        df = _users_df()
        result = df.cast_schema(RenamedUsers)
        assert isinstance(result, DataFrame)
        assert result._data.columns == ["user_id", "user_name"]
        assert result._data["user_id"].to_list() == [1, 2, 3, 4, 5]

    def test_cast_schema_missing_raises(self) -> None:
        class Bad(Schema):
            id: Column[UInt64]
            email: Column[Utf8]

        df = _users_df()
        with pytest.raises(SchemaError) as exc_info:
            df.cast_schema(Bad)
        assert "email" in exc_info.value.missing_columns


# ---------------------------------------------------------------------------
# lazy → collect roundtrip
# ---------------------------------------------------------------------------


class TestLazyCollect:
    def test_lazy_filter_collect(self) -> None:
        df = _users_df()
        lf = df.lazy()
        assert isinstance(lf, LazyFrame)
        result = lf.filter(Users.age > 30).collect()
        assert isinstance(result, DataFrame)
        assert result._data.shape == (2, 3)

    def test_lazy_select_collect(self) -> None:
        df = _users_df()
        lf = df.lazy()
        result = lf.select(Users.id, Users.name).collect()
        assert result._data.columns == ["id", "name"]


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_passes(self) -> None:
        df = _users_df()
        result = df.validate()
        assert result is df

    def test_validate_fails_missing_column(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([1], dtype=pl.UInt64),
                "name": ["Alice"],
                # missing "age"
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        with pytest.raises(SchemaError) as exc_info:
            df.validate()
        assert "age" in exc_info.value.missing_columns

    def test_validate_fails_type_mismatch(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 2],  # Int64, not UInt64
                "name": ["Alice", "Bob"],
                "age": [30, 25],  # Int64, not UInt64
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        with pytest.raises(SchemaError) as exc_info:
            df.validate()
        assert len(exc_info.value.type_mismatches) > 0


# ---------------------------------------------------------------------------
# Introspection properties
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_height(self) -> None:
        df = _users_df()
        assert df.height == 5

    def test_len(self) -> None:
        df = _users_df()
        assert len(df) == 5

    def test_width(self) -> None:
        df = _users_df()
        assert df.width == 3

    def test_shape(self) -> None:
        df = _users_df()
        assert df.shape == (5, 3)

    def test_is_empty_false(self) -> None:
        df = _users_df()
        assert df.is_empty() is False

    def test_is_empty_true(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.UInt64),
                "name": pl.Series([], dtype=pl.Utf8),
                "age": pl.Series([], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        assert df.is_empty() is True

    def test_height_after_filter(self) -> None:
        df = _users_df().filter(Users.age > 30)
        assert df.height == 2

    def test_lazyframe_width(self) -> None:
        lf = _users_df().lazy()
        assert lf.width == 3


# ---------------------------------------------------------------------------
# iter_rows_as
# ---------------------------------------------------------------------------


class TestIterRowsAs:
    def test_iter_rows_as_dict(self) -> None:
        df = _users_df()
        rows = list(df.iter_rows_as(dict))
        assert len(rows) == 5
        assert rows[0] == {"id": 1, "name": "Alice", "age": 30}

    def test_iter_rows_as_schema_row(self) -> None:
        df = _users_df()
        rows = list(df.iter_rows_as(Users.Row))
        assert len(rows) == 5
        assert rows[0].id == 1
        assert rows[0].name == "Alice"
        assert rows[0].age == 30

    def test_iter_rows_as_schema_row_type(self) -> None:
        df = _users_df()
        rows = list(df.iter_rows_as(Users.Row))
        assert type(rows[0]).__name__ == "UsersRow"

    def test_iter_rows_as_empty(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.UInt64),
                "name": pl.Series([], dtype=pl.Utf8),
                "age": pl.Series([], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        rows = list(df.iter_rows_as(Users.Row))
        assert rows == []

    def test_iter_rows_as_after_filter(self) -> None:
        df = _users_df().filter(Users.age > 35)
        rows = list(df.iter_rows_as(Users.Row))
        assert len(rows) == 1
        assert rows[0].name == "Eve"


# ---------------------------------------------------------------------------
# when/then/otherwise
# ---------------------------------------------------------------------------


class TestWhenThenOtherwise:
    def test_when_in_with_columns(self) -> None:
        df = _users_df()
        result = df.with_columns(
            when(Users.age > 30).then(Users.age).otherwise(lit(0)).alias(Users.age)
        )
        ages = result._data["age"].to_list()
        # Alice=30 → 0, Bob=25 → 0, Charlie=35 → 35, Diana=28 → 0, Eve=40 → 40
        assert ages == [0, 0, 35, 0, 40]

    def test_when_chained_branches(self) -> None:
        df = _users_df()
        result = df.with_columns(
            when(Users.age > 35)
            .then(lit("senior"))
            .when(Users.age > 27)
            .then(lit("adult"))
            .otherwise(lit("young"))
            .alias(Users.name)
        )
        names = result._data["name"].to_list()
        # Alice=30→adult, Bob=25→young, Charlie=35→adult, Diana=28→adult, Eve=40→senior
        assert names == ["adult", "young", "adult", "adult", "senior"]

    def test_when_without_otherwise_produces_null(self) -> None:
        df = _users_df()
        result = df.with_columns(when(Users.age > 35).then(lit("old")).alias(Users.name))
        names = result._data["name"].to_list()
        # Alice=30→null, Bob=25→null, Charlie=35→null, Diana=28→null, Eve=40→old
        assert names == [None, None, None, None, "old"]

    def test_when_with_complex_condition(self) -> None:
        df = _users_df()
        result = df.with_columns(
            when((Users.age > 25) & (Users.age < 35))
            .then(lit("mid"))
            .otherwise(lit("other"))
            .alias(Users.name)
        )
        names = result._data["name"].to_list()
        # Alice=30→mid, Bob=25→other, Charlie=35→other, Diana=28→mid, Eve=40→other
        assert names == ["mid", "other", "other", "mid", "other"]

    def test_when_with_expression_values(self) -> None:
        df = _users_df()
        result = df.with_columns(
            when(Users.age > 30).then(Users.age * 2).otherwise(Users.age).alias(Users.age)
        )
        ages = result._data["age"].to_list()
        # Alice=30→30, Bob=25→25, Charlie=35→70, Diana=28→28, Eve=40→80
        assert ages == [30, 25, 70, 28, 80]

    def test_when_null_in_condition_column(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
                "name": ["Alice", "Bob", "Eve"],
                "age": pl.Series([70, None, 25], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        result = df.with_columns(
            when(Users.age > 65).then(lit("senior")).otherwise(lit("other")).alias(Users.name)
        )
        names = result._data["name"].to_list()
        assert names == ["senior", "other", "other"]

    def test_when_empty_dataframe(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.UInt64),
                "name": pl.Series([], dtype=pl.Utf8),
                "age": pl.Series([], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        result = df.with_columns(
            when(Users.age > 65).then(lit("senior")).otherwise(lit("other")).alias(Users.name)
        )
        assert result._data["name"].to_list() == []

    def test_when_string_equality_condition(self) -> None:
        df = _users_df()
        result = df.with_columns(
            when(Users.name == "Alice").then(lit("found")).otherwise(lit("other")).alias(Users.name)
        )
        names = result._data["name"].to_list()
        assert names == ["found", "other", "other", "other", "other"]

    def test_multiple_when_in_with_columns(self) -> None:
        df = _users_df()
        result = df.with_columns(
            when(Users.age > 30).then(lit("old")).otherwise(lit("young")).alias(Users.name),
            when(Users.age > 30).then(Users.age * 2).otherwise(Users.age).alias(Users.age),
        )
        names = result._data["name"].to_list()
        ages = result._data["age"].to_list()
        assert names == ["young", "young", "old", "young", "old"]
        assert ages == [30, 25, 70, 28, 80]


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------


class TestConcat:
    def test_concat_two(self) -> None:
        df1 = _users_df()
        df2 = _users_df()
        result = concat(df1, df2)
        assert isinstance(result, DataFrame)
        assert result._data.shape[0] == 10
        ids = result._data["id"].to_list()
        assert ids == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

    def test_concat_preserves_row_order(self) -> None:
        """First frame's rows come before second frame's rows."""
        data_a = pl.DataFrame(
            {
                "id": pl.Series([10, 20], dtype=pl.UInt64),
                "name": ["X", "Y"],
                "age": pl.Series([1, 2], dtype=pl.UInt64),
            }
        )
        data_b = pl.DataFrame(
            {
                "id": pl.Series([30, 40], dtype=pl.UInt64),
                "name": ["Z", "W"],
                "age": pl.Series([3, 4], dtype=pl.UInt64),
            }
        )
        df_a = DataFrame(_data=data_a, _schema=Users, _backend=PolarsBackend())
        df_b = DataFrame(_data=data_b, _schema=Users, _backend=PolarsBackend())
        result = concat(df_a, df_b)
        assert result._data["id"].to_list() == [10, 20, 30, 40]
        assert result._data["name"].to_list() == ["X", "Y", "Z", "W"]

    def test_concat_three(self) -> None:
        df1 = _users_df()
        df2 = _users_df()
        df3 = _users_df()
        result = concat(df1, df2, df3)
        assert result._data.shape[0] == 15

    def test_concat_preserves_schema(self) -> None:
        df1 = _users_df()
        df2 = _users_df()
        result = concat(df1, df2)
        assert result._schema is Users

    def test_concat_result_supports_filter(self) -> None:
        """Concat result can be used in downstream operations."""
        df1 = _users_df()
        df2 = _users_df()
        result = concat(df1, df2).filter(Users.age > 30)
        ages = result._data["age"].to_list()
        assert all(a > 30 for a in ages)
        assert len(ages) == 4  # Charlie(35) + Eve(40) × 2

    def test_concat_empty_and_nonempty(self) -> None:
        df1 = _users_df()
        empty = pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.UInt64),
                "name": pl.Series([], dtype=pl.Utf8),
                "age": pl.Series([], dtype=pl.UInt64),
            }
        )
        df2 = DataFrame(_data=empty, _schema=Users, _backend=PolarsBackend())
        result = concat(df1, df2)
        assert result._data.shape[0] == 5

    def test_concat_lazyframes(self) -> None:
        df1 = _users_df().lazy()
        df2 = _users_df().lazy()
        result = concat(df1, df2)
        assert isinstance(result, LazyFrame)
        collected = result.collect()
        assert collected._data.shape[0] == 10
        ids = collected._data["id"].to_list()
        assert ids == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]


class TestItem:
    def test_item_no_arg(self) -> None:
        """Extract scalar from a 1×1 DataFrame."""
        df = _users_df()
        one_cell = df.select(Users.id).head(1)
        value = one_cell.item()
        assert value == 1
        assert isinstance(value, int)

    def test_item_with_column(self) -> None:
        """Extract a named column from a 1-row DataFrame."""
        df = _users_df().head(1)
        value = df.item(Users.name)
        assert value == "Alice"
        assert isinstance(value, str)

    def test_item_numeric_type(self) -> None:
        """item() returns a Python int, not a backend-specific type."""
        df = _users_df().head(1)
        value = df.item(Users.age)
        assert value == 30
        assert type(value) is int

    def test_item_too_many_rows_raises(self) -> None:
        df = _users_df().select(Users.id)
        with pytest.raises(ValueError, match="1.1"):
            df.item()

    def test_item_too_many_columns_raises(self) -> None:
        df = _users_df().head(1)
        with pytest.raises(ValueError, match="1.1"):
            df.item()

    def test_item_with_column_too_many_rows_raises(self) -> None:
        df = _users_df()
        with pytest.raises(ValueError, match="1 row"):
            df.item(Users.name)
