"""Unit tests for the join system — JoinCondition, JoinedDataFrame, JoinedLazyFrame."""

from __future__ import annotations

import pytest

from colnade import (
    BinOp,
    Column,
    DataFrame,
    JoinCondition,
    JoinedDataFrame,
    JoinedLazyFrame,
    LazyFrame,
    Schema,
    UInt64,
    UntypedDataFrame,
    UntypedLazyFrame,
    Utf8,
)

# ---------------------------------------------------------------------------
# Test fixture schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]


class Orders(Schema):
    id: Column[UInt64]
    user_id: Column[UInt64]
    amount: Column[UInt64]


# ---------------------------------------------------------------------------
# Minimal mock backend — returns source data unchanged for all operations
# ---------------------------------------------------------------------------


class _MockBackend:
    """Backend stub for unit tests. Returns the first positional arg (source)."""

    def __getattr__(self, name: str):  # noqa: ANN204
        def _method(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            return args[0] if args else None

        return _method


_BACKEND = _MockBackend()


# ---------------------------------------------------------------------------
# JoinCondition creation and __eq__ dispatch
# ---------------------------------------------------------------------------


class TestJoinCondition:
    def test_cross_schema_eq_returns_join_condition(self) -> None:
        result = Users.id == Orders.user_id
        assert isinstance(result, JoinCondition)

    def test_join_condition_stores_left(self) -> None:
        result = Users.id == Orders.user_id
        assert result.left is Users.id

    def test_join_condition_stores_right(self) -> None:
        result = Users.id == Orders.user_id
        assert result.right is Orders.user_id

    def test_same_schema_eq_returns_binop(self) -> None:
        result = Users.age == Users.id
        assert isinstance(result, BinOp)

    def test_same_schema_eq_not_join_condition(self) -> None:
        result = Users.age == Users.id
        assert not isinstance(result, JoinCondition)

    def test_column_vs_literal_returns_binop(self) -> None:
        result = Users.age == 25
        assert isinstance(result, BinOp)

    def test_join_condition_repr(self) -> None:
        result = Users.id == Orders.user_id
        r = repr(result)
        assert "JoinCondition" in r
        assert "id" in r
        assert "user_id" in r


# ---------------------------------------------------------------------------
# JoinedDataFrame — construction and operations
# ---------------------------------------------------------------------------


class TestJoinedDataFrame:
    def setup_method(self) -> None:
        self.users_df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)
        self.orders_df: DataFrame[Orders] = DataFrame(_schema=Orders, _backend=_BACKEND)
        self.cond = Users.id == Orders.user_id
        assert isinstance(self.cond, JoinCondition)
        self.joined = self.users_df.join(self.orders_df, on=self.cond)

    def test_join_returns_joined_dataframe(self) -> None:
        assert isinstance(self.joined, JoinedDataFrame)

    def test_joined_stores_schemas(self) -> None:
        assert self.joined._schema_left is Users
        assert self.joined._schema_right is Orders

    def test_repr(self) -> None:
        jdf = JoinedDataFrame(_schema_left=Users, _schema_right=Orders)
        assert repr(jdf) == "JoinedDataFrame[Users, Orders]"

    def test_to_native(self) -> None:
        assert self.joined.to_native() is self.joined._data

    def test_filter_returns_joined(self) -> None:
        result = self.joined.filter(Users.age > 18)
        assert isinstance(result, JoinedDataFrame)

    def test_sort_returns_joined(self) -> None:
        result = self.joined.sort(Users.name)
        assert isinstance(result, JoinedDataFrame)

    def test_sort_with_right_schema_col(self) -> None:
        result = self.joined.sort(Orders.amount)
        assert isinstance(result, JoinedDataFrame)

    def test_limit_returns_joined(self) -> None:
        result = self.joined.limit(10)
        assert isinstance(result, JoinedDataFrame)

    def test_unique_returns_joined(self) -> None:
        result = self.joined.unique(Users.id)
        assert isinstance(result, JoinedDataFrame)

    def test_drop_nulls_returns_joined(self) -> None:
        result = self.joined.drop_nulls(Users.name)
        assert isinstance(result, JoinedDataFrame)

    def test_with_columns_returns_joined(self) -> None:
        result = self.joined.with_columns(Users.age + 1)
        assert isinstance(result, JoinedDataFrame)

    def test_select_returns_dataframe(self) -> None:
        result = self.joined.select(Users.name, Orders.amount)
        assert isinstance(result, DataFrame)
        assert result._schema is None

    def test_select_single_column(self) -> None:
        result = self.joined.select(Users.name)
        assert isinstance(result, DataFrame)


# ---------------------------------------------------------------------------
# JoinedDataFrame — conversions
# ---------------------------------------------------------------------------


class TestJoinedDataFrameConversions:
    def setup_method(self) -> None:
        self.joined = JoinedDataFrame(_schema_left=Users, _schema_right=Orders, _backend=_BACKEND)

    def test_lazy_returns_joined_lazyframe(self) -> None:
        result = self.joined.lazy()
        assert isinstance(result, JoinedLazyFrame)
        assert result._schema_left is Users
        assert result._schema_right is Orders

    def test_untyped_returns_untyped(self) -> None:
        result = self.joined.untyped()
        assert isinstance(result, UntypedDataFrame)

    def test_untyped_preserves_backend(self) -> None:
        result = self.joined.untyped()
        assert result._backend is _BACKEND


# ---------------------------------------------------------------------------
# JoinedLazyFrame — construction and operations
# ---------------------------------------------------------------------------


class TestJoinedLazyFrame:
    def setup_method(self) -> None:
        self.users_lf: LazyFrame[Users] = LazyFrame(_schema=Users, _backend=_BACKEND)
        self.orders_lf: LazyFrame[Orders] = LazyFrame(_schema=Orders, _backend=_BACKEND)
        self.cond = Users.id == Orders.user_id
        assert isinstance(self.cond, JoinCondition)
        self.joined = self.users_lf.join(self.orders_lf, on=self.cond)

    def test_join_returns_joined_lazyframe(self) -> None:
        assert isinstance(self.joined, JoinedLazyFrame)

    def test_joined_stores_schemas(self) -> None:
        assert self.joined._schema_left is Users
        assert self.joined._schema_right is Orders

    def test_repr(self) -> None:
        jlf = JoinedLazyFrame(_schema_left=Users, _schema_right=Orders)
        assert repr(jlf) == "JoinedLazyFrame[Users, Orders]"

    def test_to_native(self) -> None:
        assert self.joined.to_native() is self.joined._data

    def test_filter_returns_joined_lazy(self) -> None:
        result = self.joined.filter(Users.age > 18)
        assert isinstance(result, JoinedLazyFrame)

    def test_sort_returns_joined_lazy(self) -> None:
        result = self.joined.sort(Users.name)
        assert isinstance(result, JoinedLazyFrame)

    def test_limit_returns_joined_lazy(self) -> None:
        result = self.joined.limit(10)
        assert isinstance(result, JoinedLazyFrame)

    def test_unique_returns_joined_lazy(self) -> None:
        result = self.joined.unique(Users.id)
        assert isinstance(result, JoinedLazyFrame)

    def test_drop_nulls_returns_joined_lazy(self) -> None:
        result = self.joined.drop_nulls(Users.name)
        assert isinstance(result, JoinedLazyFrame)

    def test_with_columns_returns_joined_lazy(self) -> None:
        result = self.joined.with_columns(Users.age + 1)
        assert isinstance(result, JoinedLazyFrame)

    def test_select_returns_lazyframe(self) -> None:
        result = self.joined.select(Users.name, Orders.amount)
        assert isinstance(result, LazyFrame)
        assert result._schema is None

    def test_collect_returns_joined_dataframe(self) -> None:
        result = self.joined.collect()
        assert isinstance(result, JoinedDataFrame)
        assert result._schema_left is Users
        assert result._schema_right is Orders

    def test_untyped_returns_untyped_lazy(self) -> None:
        result = self.joined.untyped()
        assert isinstance(result, UntypedLazyFrame)

    def test_untyped_preserves_backend(self) -> None:
        result = self.joined.untyped()
        assert result._backend is _BACKEND


# ---------------------------------------------------------------------------
# Joined frame restrictions — methods that require cast_schema first
# ---------------------------------------------------------------------------


class TestJoinedDataFrameRestrictions:
    def test_no_head(self) -> None:
        assert not hasattr(JoinedDataFrame, "head")

    def test_no_tail(self) -> None:
        assert not hasattr(JoinedDataFrame, "tail")

    def test_no_sample(self) -> None:
        assert not hasattr(JoinedDataFrame, "sample")

    def test_no_group_by(self) -> None:
        assert not hasattr(JoinedDataFrame, "group_by")


class TestJoinedLazyFrameRestrictions:
    def test_no_head(self) -> None:
        assert not hasattr(JoinedLazyFrame, "head")

    def test_no_tail(self) -> None:
        assert not hasattr(JoinedLazyFrame, "tail")

    def test_no_sample(self) -> None:
        assert not hasattr(JoinedLazyFrame, "sample")

    def test_no_group_by(self) -> None:
        assert not hasattr(JoinedLazyFrame, "group_by")


# ---------------------------------------------------------------------------
# Join how parameter
# ---------------------------------------------------------------------------


class TestJoinHow:
    def setup_method(self) -> None:
        self.users_df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)
        self.orders_df: DataFrame[Orders] = DataFrame(_schema=Orders, _backend=_BACKEND)
        self.cond = Users.id == Orders.user_id
        assert isinstance(self.cond, JoinCondition)

    def test_inner(self) -> None:
        result = self.users_df.join(self.orders_df, on=self.cond, how="inner")
        assert isinstance(result, JoinedDataFrame)

    def test_left(self) -> None:
        result = self.users_df.join(self.orders_df, on=self.cond, how="left")
        assert isinstance(result, JoinedDataFrame)

    def test_outer(self) -> None:
        result = self.users_df.join(self.orders_df, on=self.cond, how="outer")
        assert isinstance(result, JoinedDataFrame)

    def test_cross(self) -> None:
        result = self.users_df.join(self.orders_df, on=self.cond, how="cross")
        assert isinstance(result, JoinedDataFrame)

    def test_default_is_inner(self) -> None:
        result = self.users_df.join(self.orders_df, on=self.cond)
        assert isinstance(result, JoinedDataFrame)


# ---------------------------------------------------------------------------
# No-backend RuntimeError tests for joined frames
# ---------------------------------------------------------------------------


class TestJoinedNoBackendRaises:
    def test_joined_dataframe_filter(self) -> None:
        jdf = JoinedDataFrame(_schema_left=Users, _schema_right=Orders)
        with pytest.raises(RuntimeError, match="requires a backend"):
            jdf.filter(Users.age > 18)

    def test_joined_dataframe_lazy(self) -> None:
        jdf = JoinedDataFrame(_schema_left=Users, _schema_right=Orders)
        with pytest.raises(RuntimeError, match="requires a backend"):
            jdf.lazy()

    def test_joined_lazyframe_collect(self) -> None:
        jlf = JoinedLazyFrame(_schema_left=Users, _schema_right=Orders)
        with pytest.raises(RuntimeError, match="requires a backend"):
            jlf.collect()

    def test_joined_lazyframe_select(self) -> None:
        jlf = JoinedLazyFrame(_schema_left=Users, _schema_right=Orders)
        with pytest.raises(RuntimeError, match="requires a backend"):
            jlf.select(Users.name)

    def test_join_without_backend(self) -> None:
        df = DataFrame(_schema=Users)
        other = DataFrame(_schema=Orders)
        cond = Users.id == Orders.user_id
        assert isinstance(cond, JoinCondition)
        with pytest.raises(RuntimeError, match="requires a backend"):
            df.join(other, on=cond)
