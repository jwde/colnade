"""Unit tests for DataFrame[S], LazyFrame[S], GroupBy, and untyped escape hatches."""

from __future__ import annotations

from colnade import (
    AliasedExpr,
    Bool,
    Column,
    DataFrame,
    Expr,
    GroupBy,
    LazyFrame,
    LazyGroupBy,
    Schema,
    UInt8,
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
    age: Column[UInt8]


class AgeStats(Schema):
    age: Column[UInt8]
    count: Column[UInt64]


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------


class TestDataFrameConstruction:
    def test_default_construction(self) -> None:
        df: DataFrame[Users] = DataFrame()
        assert isinstance(df, DataFrame)

    def test_schema_stored(self) -> None:
        df = DataFrame(_schema=Users)
        assert df._schema is Users

    def test_data_stored(self) -> None:
        sentinel = object()
        df = DataFrame(_data=sentinel)
        assert df._data is sentinel

    def test_repr_with_schema(self) -> None:
        df = DataFrame(_schema=Users)
        assert repr(df) == "DataFrame[Users]"

    def test_repr_without_schema(self) -> None:
        df = DataFrame()
        assert repr(df) == "DataFrame[Any]"


# ---------------------------------------------------------------------------
# Schema-preserving operations
# ---------------------------------------------------------------------------


class TestSchemaPreservingOps:
    def setup_method(self) -> None:
        self.df: DataFrame[Users] = DataFrame(_schema=Users)

    def test_filter_returns_dataframe(self) -> None:
        predicate: Expr[Bool] = Users.age > 18
        result = self.df.filter(predicate)
        assert isinstance(result, DataFrame)
        assert result._schema is Users

    def test_sort_returns_dataframe(self) -> None:
        result = self.df.sort(Users.name)
        assert isinstance(result, DataFrame)
        assert result._schema is Users

    def test_sort_with_sort_expr(self) -> None:
        result = self.df.sort(Users.name.desc())
        assert isinstance(result, DataFrame)

    def test_sort_descending_kwarg(self) -> None:
        result = self.df.sort(Users.name, descending=True)
        assert isinstance(result, DataFrame)

    def test_limit_returns_dataframe(self) -> None:
        result = self.df.limit(10)
        assert isinstance(result, DataFrame)
        assert result._schema is Users

    def test_head_returns_dataframe(self) -> None:
        result = self.df.head()
        assert isinstance(result, DataFrame)
        assert result._schema is Users

    def test_head_with_n(self) -> None:
        result = self.df.head(10)
        assert isinstance(result, DataFrame)

    def test_tail_returns_dataframe(self) -> None:
        result = self.df.tail()
        assert isinstance(result, DataFrame)
        assert result._schema is Users

    def test_tail_with_n(self) -> None:
        result = self.df.tail(10)
        assert isinstance(result, DataFrame)

    def test_sample_returns_dataframe(self) -> None:
        result = self.df.sample(5)
        assert isinstance(result, DataFrame)
        assert result._schema is Users

    def test_unique_returns_dataframe(self) -> None:
        result = self.df.unique(Users.name)
        assert isinstance(result, DataFrame)
        assert result._schema is Users

    def test_unique_multiple_columns(self) -> None:
        result = self.df.unique(Users.name, Users.age)
        assert isinstance(result, DataFrame)

    def test_drop_nulls_returns_dataframe(self) -> None:
        result = self.df.drop_nulls(Users.name)
        assert isinstance(result, DataFrame)
        assert result._schema is Users

    def test_with_columns_returns_dataframe(self) -> None:
        expr = Users.age + 1
        result = self.df.with_columns(expr)
        assert isinstance(result, DataFrame)
        assert result._schema is Users

    def test_with_columns_aliased(self) -> None:
        aliased: AliasedExpr[UInt8] = (Users.age + 1).alias("age_plus_one")
        result = self.df.with_columns(aliased)
        assert isinstance(result, DataFrame)


# ---------------------------------------------------------------------------
# Schema-transforming operations
# ---------------------------------------------------------------------------


class TestSchemaTransformingOps:
    def setup_method(self) -> None:
        self.df: DataFrame[Users] = DataFrame(_schema=Users)

    def test_select_single_column(self) -> None:
        result = self.df.select(Users.name)
        assert isinstance(result, DataFrame)
        assert result._schema is None

    def test_select_multiple_columns(self) -> None:
        result = self.df.select(Users.name, Users.age)
        assert isinstance(result, DataFrame)
        assert result._schema is None

    def test_select_all_columns(self) -> None:
        result = self.df.select(Users.id, Users.name, Users.age)
        assert isinstance(result, DataFrame)

    def test_group_by_agg(self) -> None:
        result = self.df.group_by(Users.age).agg(Users.id.count().alias("count"))
        assert isinstance(result, DataFrame)
        assert result._schema is None


# ---------------------------------------------------------------------------
# GroupBy
# ---------------------------------------------------------------------------


class TestGroupBy:
    def test_group_by_returns_groupby(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users)
        gb = df.group_by(Users.age)
        assert isinstance(gb, GroupBy)

    def test_group_by_keys_stored(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users)
        gb = df.group_by(Users.age, Users.name)
        assert gb._keys == (Users.age, Users.name)

    def test_group_by_schema_stored(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users)
        gb = df.group_by(Users.age)
        assert gb._schema is Users

    def test_agg_returns_dataframe(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users)
        result = df.group_by(Users.age).agg(Users.id.count().alias("count"))
        assert isinstance(result, DataFrame)
        assert result._schema is None

    def test_agg_multiple_exprs(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users)
        result = df.group_by(Users.age).agg(
            Users.id.count().alias("count"),
            Users.name.count().alias("name_count"),
        )
        assert isinstance(result, DataFrame)


# ---------------------------------------------------------------------------
# Conversion: lazy, untyped, collect
# ---------------------------------------------------------------------------


class TestConversion:
    def test_lazy_returns_lazyframe(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users)
        result = df.lazy()
        assert isinstance(result, LazyFrame)
        assert result._schema is Users

    def test_untyped_returns_untyped(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users)
        result = df.untyped()
        assert isinstance(result, UntypedDataFrame)

    def test_collect_returns_dataframe(self) -> None:
        lf: LazyFrame[Users] = LazyFrame(_schema=Users)
        result = lf.collect()
        assert isinstance(result, DataFrame)
        assert result._schema is Users

    def test_validate_returns_self(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users)
        result = df.validate()
        assert result is df

    def test_to_batches_not_implemented(self) -> None:
        import pytest

        df: DataFrame[Users] = DataFrame(_schema=Users)
        with pytest.raises(NotImplementedError):
            df.to_batches()


# ---------------------------------------------------------------------------
# LazyFrame operations
# ---------------------------------------------------------------------------


class TestLazyFrame:
    def setup_method(self) -> None:
        self.lf: LazyFrame[Users] = LazyFrame(_schema=Users)

    def test_repr_with_schema(self) -> None:
        assert repr(self.lf) == "LazyFrame[Users]"

    def test_repr_without_schema(self) -> None:
        lf = LazyFrame()
        assert repr(lf) == "LazyFrame[Any]"

    def test_filter_returns_lazyframe(self) -> None:
        result = self.lf.filter(Users.age > 18)
        assert isinstance(result, LazyFrame)
        assert result._schema is Users

    def test_sort_returns_lazyframe(self) -> None:
        result = self.lf.sort(Users.name)
        assert isinstance(result, LazyFrame)
        assert result._schema is Users

    def test_limit_returns_lazyframe(self) -> None:
        result = self.lf.limit(10)
        assert isinstance(result, LazyFrame)
        assert result._schema is Users

    def test_unique_returns_lazyframe(self) -> None:
        result = self.lf.unique(Users.name)
        assert isinstance(result, LazyFrame)

    def test_drop_nulls_returns_lazyframe(self) -> None:
        result = self.lf.drop_nulls(Users.name)
        assert isinstance(result, LazyFrame)

    def test_with_columns_returns_lazyframe(self) -> None:
        result = self.lf.with_columns(Users.age + 1)
        assert isinstance(result, LazyFrame)

    def test_select_returns_lazyframe(self) -> None:
        result = self.lf.select(Users.name, Users.age)
        assert isinstance(result, LazyFrame)
        assert result._schema is None

    def test_group_by_returns_lazy_groupby(self) -> None:
        gb = self.lf.group_by(Users.age)
        assert isinstance(gb, LazyGroupBy)

    def test_collect_returns_dataframe(self) -> None:
        result = self.lf.collect()
        assert isinstance(result, DataFrame)
        assert result._schema is Users

    def test_untyped_returns_untyped_lazy(self) -> None:
        result = self.lf.untyped()
        assert isinstance(result, UntypedLazyFrame)

    def test_validate_returns_self(self) -> None:
        result = self.lf.validate()
        assert result is self.lf


# ---------------------------------------------------------------------------
# LazyFrame restrictions (no head, tail, sample)
# ---------------------------------------------------------------------------


class TestLazyFrameRestrictions:
    def test_no_head(self) -> None:
        assert not hasattr(LazyFrame, "head")

    def test_no_tail(self) -> None:
        assert not hasattr(LazyFrame, "tail")

    def test_no_sample(self) -> None:
        assert not hasattr(LazyFrame, "sample")

    def test_no_to_batches(self) -> None:
        assert not hasattr(LazyFrame, "to_batches")


# ---------------------------------------------------------------------------
# LazyGroupBy
# ---------------------------------------------------------------------------


class TestLazyGroupBy:
    def test_lazy_groupby_agg_returns_lazyframe(self) -> None:
        lf: LazyFrame[Users] = LazyFrame(_schema=Users)
        result = lf.group_by(Users.age).agg(Users.id.count().alias("count"))
        assert isinstance(result, LazyFrame)
        assert result._schema is None

    def test_lazy_groupby_keys_stored(self) -> None:
        lf: LazyFrame[Users] = LazyFrame(_schema=Users)
        gb = lf.group_by(Users.age, Users.name)
        assert gb._keys == (Users.age, Users.name)


# ---------------------------------------------------------------------------
# UntypedDataFrame
# ---------------------------------------------------------------------------


class TestUntypedDataFrame:
    def setup_method(self) -> None:
        self.udf = UntypedDataFrame()

    def test_select_returns_untyped(self) -> None:
        result = self.udf.select("name", "age")
        assert isinstance(result, UntypedDataFrame)

    def test_filter_returns_untyped(self) -> None:
        result = self.udf.filter("age > 18")
        assert isinstance(result, UntypedDataFrame)

    def test_with_columns_returns_untyped(self) -> None:
        result = self.udf.with_columns("something")
        assert isinstance(result, UntypedDataFrame)

    def test_sort_returns_untyped(self) -> None:
        result = self.udf.sort("name")
        assert isinstance(result, UntypedDataFrame)

    def test_limit_returns_untyped(self) -> None:
        result = self.udf.limit(10)
        assert isinstance(result, UntypedDataFrame)

    def test_head_returns_untyped(self) -> None:
        result = self.udf.head()
        assert isinstance(result, UntypedDataFrame)

    def test_tail_returns_untyped(self) -> None:
        result = self.udf.tail()
        assert isinstance(result, UntypedDataFrame)

    def test_to_typed_returns_dataframe(self) -> None:
        result = self.udf.to_typed(Users)
        assert isinstance(result, DataFrame)
        assert result._schema is Users


# ---------------------------------------------------------------------------
# UntypedLazyFrame
# ---------------------------------------------------------------------------


class TestUntypedLazyFrame:
    def setup_method(self) -> None:
        self.ulf = UntypedLazyFrame()

    def test_select_returns_untyped_lazy(self) -> None:
        result = self.ulf.select("name")
        assert isinstance(result, UntypedLazyFrame)

    def test_filter_returns_untyped_lazy(self) -> None:
        result = self.ulf.filter("age > 18")
        assert isinstance(result, UntypedLazyFrame)

    def test_with_columns_returns_untyped_lazy(self) -> None:
        result = self.ulf.with_columns("something")
        assert isinstance(result, UntypedLazyFrame)

    def test_sort_returns_untyped_lazy(self) -> None:
        result = self.ulf.sort("name")
        assert isinstance(result, UntypedLazyFrame)

    def test_limit_returns_untyped_lazy(self) -> None:
        result = self.ulf.limit(10)
        assert isinstance(result, UntypedLazyFrame)

    def test_collect_returns_untyped_dataframe(self) -> None:
        result = self.ulf.collect()
        assert isinstance(result, UntypedDataFrame)

    def test_to_typed_returns_lazyframe(self) -> None:
        result = self.ulf.to_typed(Users)
        assert isinstance(result, LazyFrame)
        assert result._schema is Users
