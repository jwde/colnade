"""Integration tests for runtime cross-schema expression guard (issue #80).

When validation is enabled, DataFrame operations reject expressions that
reference columns not in the frame's schema.
"""

from __future__ import annotations

import polars as pl
import pytest

import colnade
from colnade import Column, DataFrame, LazyFrame, Schema, SchemaError, UInt64, Utf8, ValidationLevel
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

backend = PolarsBackend()


def _users_df() -> DataFrame[Users]:
    data = pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "name": ["Alice", "Bob", "Charlie"],
            "age": pl.Series([30, 25, 35], dtype=pl.UInt64),
        }
    )
    return DataFrame(_data=data, _schema=Users, _backend=backend)


def _users_lazy() -> LazyFrame[Users]:
    data = pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "name": ["Alice", "Bob", "Charlie"],
            "age": pl.Series([30, 25, 35], dtype=pl.UInt64),
        }
    ).lazy()
    return LazyFrame(_data=data, _schema=Users, _backend=backend)


def _joined_eager():
    users_data = pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "name": ["Alice", "Bob", "Charlie"],
            "age": pl.Series([30, 25, 35], dtype=pl.UInt64),
        }
    )
    orders_data = pl.DataFrame(
        {
            "order_id": pl.Series([10, 20, 30], dtype=pl.UInt64),
            "user_id": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "amount": pl.Series([100, 200, 300], dtype=pl.UInt64),
        }
    )
    users = DataFrame(_data=users_data, _schema=Users, _backend=backend)
    orders = DataFrame(_data=orders_data, _schema=Orders, _backend=backend)
    return users.join(orders, on=Users.id == Orders.user_id)  # type: ignore[arg-type]


def _joined_lazy():
    users_data = pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "name": ["Alice", "Bob", "Charlie"],
            "age": pl.Series([30, 25, 35], dtype=pl.UInt64),
        }
    ).lazy()
    orders_data = pl.DataFrame(
        {
            "order_id": pl.Series([10, 20, 30], dtype=pl.UInt64),
            "user_id": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "amount": pl.Series([100, 200, 300], dtype=pl.UInt64),
        }
    ).lazy()
    users = LazyFrame(_data=users_data, _schema=Users, _backend=backend)
    orders = LazyFrame(_data=orders_data, _schema=Orders, _backend=backend)
    return users.join(orders, on=Users.id == Orders.user_id)  # type: ignore[arg-type]


class Other(Schema):
    foo: Column[UInt64]


@pytest.fixture(autouse=True)
def _enable_validation():
    """Enable structural validation for all tests, restore after."""
    colnade.set_validation(ValidationLevel.STRUCTURAL)
    yield
    colnade.set_validation(ValidationLevel.OFF)


# ---------------------------------------------------------------------------
# DataFrame — wrong schema raises SchemaError
# ---------------------------------------------------------------------------


class TestDataFrameCrossSchemaGuard:
    def test_filter_wrong_schema(self) -> None:
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.filter(Orders.amount > 100)

    def test_sort_wrong_schema_column(self) -> None:
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.sort(Orders.amount)

    def test_sort_wrong_schema_sort_expr(self) -> None:
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.sort(Orders.amount.desc())

    def test_with_columns_wrong_schema(self) -> None:
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.with_columns((Orders.amount * 2).alias(Users.age))

    def test_select_wrong_schema(self) -> None:
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.select(Users.id, Orders.amount)

    def test_unique_wrong_schema(self) -> None:
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.unique(Orders.amount)

    def test_drop_nulls_wrong_schema(self) -> None:
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.drop_nulls(Orders.amount)

    def test_group_by_wrong_schema(self) -> None:
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.group_by(Orders.amount)

    def test_group_by_agg_wrong_schema(self) -> None:
        df = _users_df()
        gb = df.group_by(Users.name)
        with pytest.raises(SchemaError, match="amount"):
            gb.agg(Orders.amount.sum().alias(Users.age))


# ---------------------------------------------------------------------------
# DataFrame — correct schema passes
# ---------------------------------------------------------------------------


class TestDataFrameCorrectSchema:
    def test_filter_correct(self) -> None:
        df = _users_df()
        result = df.filter(Users.age > 25)
        assert result.height == 2

    def test_sort_correct(self) -> None:
        df = _users_df()
        result = df.sort(Users.age)
        assert result.height == 3

    def test_with_columns_correct(self) -> None:
        df = _users_df()
        result = df.with_columns((Users.age + 1).alias(Users.age))
        assert result.height == 3

    def test_select_correct(self) -> None:
        df = _users_df()
        result = df.select(Users.id, Users.name)
        assert result.to_native().columns == ["id", "name"]

    def test_unique_correct(self) -> None:
        df = _users_df()
        result = df.unique(Users.name)
        assert result.height == 3

    def test_drop_nulls_correct(self) -> None:
        df = _users_df()
        result = df.drop_nulls(Users.age)
        assert result.height == 3

    def test_group_by_agg_correct(self) -> None:
        df = _users_df()
        result = df.group_by(Users.name).agg(Users.age.sum().alias(Users.age))
        assert result.to_native().height == 3


# ---------------------------------------------------------------------------
# LazyFrame — wrong schema raises SchemaError
# ---------------------------------------------------------------------------


class TestLazyFrameCrossSchemaGuard:
    def test_filter_wrong_schema(self) -> None:
        lf = _users_lazy()
        with pytest.raises(SchemaError, match="amount"):
            lf.filter(Orders.amount > 100)

    def test_sort_wrong_schema(self) -> None:
        lf = _users_lazy()
        with pytest.raises(SchemaError, match="amount"):
            lf.sort(Orders.amount)

    def test_with_columns_wrong_schema(self) -> None:
        lf = _users_lazy()
        with pytest.raises(SchemaError, match="amount"):
            lf.with_columns((Orders.amount * 2).alias(Users.age))

    def test_select_wrong_schema(self) -> None:
        lf = _users_lazy()
        with pytest.raises(SchemaError, match="amount"):
            lf.select(Orders.amount)

    def test_unique_wrong_schema(self) -> None:
        lf = _users_lazy()
        with pytest.raises(SchemaError, match="amount"):
            lf.unique(Orders.amount)

    def test_drop_nulls_wrong_schema(self) -> None:
        lf = _users_lazy()
        with pytest.raises(SchemaError, match="amount"):
            lf.drop_nulls(Orders.amount)

    def test_group_by_wrong_schema(self) -> None:
        lf = _users_lazy()
        with pytest.raises(SchemaError, match="amount"):
            lf.group_by(Orders.amount)

    def test_group_by_agg_wrong_schema(self) -> None:
        lf = _users_lazy()
        gb = lf.group_by(Users.name)
        with pytest.raises(SchemaError, match="amount"):
            gb.agg(Orders.amount.sum().alias(Users.age))


# ---------------------------------------------------------------------------
# LazyFrame — correct schema passes
# ---------------------------------------------------------------------------


class TestLazyFrameCorrectSchema:
    def test_filter_correct(self) -> None:
        lf = _users_lazy()
        result = lf.filter(Users.age > 25).collect()
        assert result.height == 2

    def test_sort_correct(self) -> None:
        lf = _users_lazy()
        result = lf.sort(Users.age).collect()
        assert result.height == 3

    def test_with_columns_correct(self) -> None:
        lf = _users_lazy()
        result = lf.with_columns((Users.age + 1).alias(Users.age)).collect()
        assert result.height == 3

    def test_select_correct(self) -> None:
        lf = _users_lazy()
        result = lf.select(Users.id, Users.name).collect()
        assert result.to_native().columns == ["id", "name"]

    def test_unique_correct(self) -> None:
        lf = _users_lazy()
        result = lf.unique(Users.name).collect()
        assert result.height == 3

    def test_drop_nulls_correct(self) -> None:
        lf = _users_lazy()
        result = lf.drop_nulls(Users.age).collect()
        assert result.height == 3

    def test_group_by_agg_correct(self) -> None:
        lf = _users_lazy()
        result = lf.group_by(Users.name).agg(Users.age.sum().alias(Users.age)).collect()
        assert result.to_native().height == 3


# ---------------------------------------------------------------------------
# Validation level coverage — fires at STRUCTURAL and FULL, not OFF
# ---------------------------------------------------------------------------


class TestValidationLevels:
    def test_off_does_not_raise_schema_error(self) -> None:
        colnade.set_validation(ValidationLevel.OFF)
        df = _users_df()
        # Guard doesn't fire — Polars will raise its own error
        try:
            df.filter(Orders.amount > 100)
        except SchemaError:
            pytest.fail("SchemaError should not be raised when validation is OFF")
        except Exception:
            pass  # Expected — Polars error for missing column

    def test_structural_raises(self) -> None:
        colnade.set_validation(ValidationLevel.STRUCTURAL)
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.filter(Orders.amount > 100)

    def test_full_raises(self) -> None:
        colnade.set_validation(ValidationLevel.FULL)
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.filter(Orders.amount > 100)

    def test_off_lazy_does_not_raise_schema_error(self) -> None:
        colnade.set_validation(ValidationLevel.OFF)
        lf = _users_lazy()
        try:
            lf.filter(Orders.amount > 100)
        except SchemaError:
            pytest.fail("SchemaError should not be raised when validation is OFF")
        except Exception:
            pass

    def test_full_lazy_raises(self) -> None:
        colnade.set_validation(ValidationLevel.FULL)
        lf = _users_lazy()
        with pytest.raises(SchemaError, match="amount"):
            lf.filter(Orders.amount > 100)


# ---------------------------------------------------------------------------
# JoinedDataFrame — accepts columns from both schemas, rejects unknown
# ---------------------------------------------------------------------------


class TestJoinedDataFrameCrossSchemaGuard:
    # --- Positive: both schemas accepted ---

    def test_filter_left_schema_ok(self) -> None:
        joined = _joined_eager()
        result = joined.filter(Users.age > 25)
        assert result.to_native().height == 2

    def test_filter_right_schema_ok(self) -> None:
        joined = _joined_eager()
        result = joined.filter(Orders.amount > 150)
        assert result.to_native().height == 2

    def test_sort_both_schemas_ok(self) -> None:
        joined = _joined_eager()
        result = joined.sort(Users.age)
        assert result.to_native().height == 3

    def test_select_both_schemas_ok(self) -> None:
        joined = _joined_eager()
        result = joined.select(Users.name, Orders.amount)
        assert set(result.to_native().columns) == {"name", "amount"}

    def test_with_columns_both_schemas_ok(self) -> None:
        joined = _joined_eager()
        result = joined.with_columns((Orders.amount + Users.age).alias(Orders.amount))
        assert result.to_native().height == 3

    def test_unique_both_schemas_ok(self) -> None:
        joined = _joined_eager()
        result = joined.unique(Users.name)
        assert result.to_native().height == 3

    def test_drop_nulls_both_schemas_ok(self) -> None:
        joined = _joined_eager()
        result = joined.drop_nulls(Orders.amount)
        assert result.to_native().height == 3

    # --- Negative: unknown third schema rejected ---

    def test_filter_unknown_schema_raises(self) -> None:
        joined = _joined_eager()
        with pytest.raises(SchemaError, match="foo"):
            joined.filter(Other.foo > 1)

    def test_sort_unknown_schema_raises(self) -> None:
        joined = _joined_eager()
        with pytest.raises(SchemaError, match="foo"):
            joined.sort(Other.foo)

    def test_select_unknown_schema_raises(self) -> None:
        joined = _joined_eager()
        with pytest.raises(SchemaError, match="foo"):
            joined.select(Users.name, Other.foo)

    def test_with_columns_unknown_schema_raises(self) -> None:
        joined = _joined_eager()
        with pytest.raises(SchemaError, match="foo"):
            joined.with_columns((Other.foo * 2).alias(Users.age))

    def test_unique_unknown_schema_raises(self) -> None:
        joined = _joined_eager()
        with pytest.raises(SchemaError, match="foo"):
            joined.unique(Other.foo)

    def test_drop_nulls_unknown_schema_raises(self) -> None:
        joined = _joined_eager()
        with pytest.raises(SchemaError, match="foo"):
            joined.drop_nulls(Other.foo)


# ---------------------------------------------------------------------------
# JoinedLazyFrame — same coverage as JoinedDataFrame
# ---------------------------------------------------------------------------


class TestJoinedLazyFrameCrossSchemaGuard:
    # --- Positive: both schemas accepted ---

    def test_filter_left_schema_ok(self) -> None:
        joined = _joined_lazy()
        result = joined.filter(Users.age > 25).collect()
        assert result.to_native().height == 2

    def test_filter_right_schema_ok(self) -> None:
        joined = _joined_lazy()
        result = joined.filter(Orders.amount > 150).collect()
        assert result.to_native().height == 2

    def test_sort_ok(self) -> None:
        joined = _joined_lazy()
        result = joined.sort(Users.age).collect()
        assert result.to_native().height == 3

    def test_select_both_schemas_ok(self) -> None:
        joined = _joined_lazy()
        result = joined.select(Users.name, Orders.amount).collect()
        assert set(result.to_native().columns) == {"name", "amount"}

    def test_with_columns_ok(self) -> None:
        joined = _joined_lazy()
        result = joined.with_columns((Orders.amount + Users.age).alias(Orders.amount)).collect()
        assert result.to_native().height == 3

    def test_unique_ok(self) -> None:
        joined = _joined_lazy()
        result = joined.unique(Users.name).collect()
        assert result.to_native().height == 3

    def test_drop_nulls_ok(self) -> None:
        joined = _joined_lazy()
        result = joined.drop_nulls(Orders.amount).collect()
        assert result.to_native().height == 3

    # --- Negative: unknown schema rejected ---

    def test_filter_unknown_schema_raises(self) -> None:
        joined = _joined_lazy()
        with pytest.raises(SchemaError, match="foo"):
            joined.filter(Other.foo > 1)

    def test_sort_unknown_schema_raises(self) -> None:
        joined = _joined_lazy()
        with pytest.raises(SchemaError, match="foo"):
            joined.sort(Other.foo)

    def test_select_unknown_schema_raises(self) -> None:
        joined = _joined_lazy()
        with pytest.raises(SchemaError, match="foo"):
            joined.select(Users.name, Other.foo)

    def test_with_columns_unknown_schema_raises(self) -> None:
        joined = _joined_lazy()
        with pytest.raises(SchemaError, match="foo"):
            joined.with_columns((Other.foo * 2).alias(Users.age))

    def test_unique_unknown_schema_raises(self) -> None:
        joined = _joined_lazy()
        with pytest.raises(SchemaError, match="foo"):
            joined.unique(Other.foo)

    def test_drop_nulls_unknown_schema_raises(self) -> None:
        joined = _joined_lazy()
        with pytest.raises(SchemaError, match="foo"):
            joined.drop_nulls(Other.foo)


# ---------------------------------------------------------------------------
# Complex expressions
# ---------------------------------------------------------------------------


class TestComplexExpressions:
    def test_logical_and_one_wrong(self) -> None:
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.filter((Users.age > 25) & (Orders.amount > 100))

    def test_nested_arithmetic_wrong(self) -> None:
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.with_columns((Users.age + Orders.amount).alias(Users.age))

    def test_fill_null_with_wrong_column(self) -> None:
        df = _users_df()
        with pytest.raises(SchemaError, match="amount"):
            df.filter(Orders.amount.is_null())
