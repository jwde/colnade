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


# ---------------------------------------------------------------------------
# Validation disabled — no error
# ---------------------------------------------------------------------------


class TestValidationDisabled:
    def test_filter_wrong_schema_no_error(self) -> None:
        colnade.set_validation(ValidationLevel.OFF)
        df = _users_df()
        # Would fail at Polars level, but the validation guard doesn't fire
        # so we only check that SchemaError is NOT raised by our guard.
        # Polars will raise its own error for the missing column.
        with pytest.raises(Exception, match="amount"):
            df.filter(Orders.amount > 100)
        # Verify the error is NOT a SchemaError (it's a Polars error)
        try:
            df.filter(Orders.amount > 100)
        except SchemaError:
            pytest.fail("SchemaError should not be raised when validation is OFF")
        except Exception:
            pass  # Expected — Polars error


# ---------------------------------------------------------------------------
# JoinedDataFrame — accepts columns from both schemas
# ---------------------------------------------------------------------------


class TestJoinedDataFrameCrossSchemaGuard:
    def _joined(self):
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

    def test_filter_left_schema_ok(self) -> None:
        joined = self._joined()
        result = joined.filter(Users.age > 25)
        assert result.to_native().height == 2

    def test_filter_right_schema_ok(self) -> None:
        joined = self._joined()
        result = joined.filter(Orders.amount > 150)
        assert result.to_native().height == 2

    def test_filter_unknown_schema_raises(self) -> None:
        class Other(Schema):
            foo: Column[UInt64]

        joined = self._joined()
        with pytest.raises(SchemaError, match="foo"):
            joined.filter(Other.foo > 1)

    def test_sort_both_schemas_ok(self) -> None:
        joined = self._joined()
        result = joined.sort(Users.age)
        assert result.to_native().height == 3

    def test_select_both_schemas_ok(self) -> None:
        joined = self._joined()
        result = joined.select(Users.name, Orders.amount)
        assert set(result.to_native().columns) == {"name", "amount"}

    def test_with_columns_both_schemas_ok(self) -> None:
        joined = self._joined()
        result = joined.with_columns((Orders.amount + Users.age).alias(Orders.amount))
        assert result.to_native().height == 3


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
