"""Unit tests for DataFrame[S], LazyFrame[S], and GroupBy."""

from __future__ import annotations

import pytest

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
    Utf8,
    concat,
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
# Minimal mock backend — returns source data unchanged for all operations
# ---------------------------------------------------------------------------


class _MockBackend:
    """Backend stub for unit tests. Returns the first positional arg (source)."""

    def row_count(self, source: object) -> int:  # noqa: ANN001
        return 0

    def iter_row_dicts(self, source: object) -> list[dict[str, object]]:  # noqa: ANN001
        return []

    def item(self, source: object, column: str | None = None) -> object:  # noqa: ANN001
        return ("item_called", source, column)

    def __getattr__(self, name: str):  # noqa: ANN204
        def _method(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            return args[0] if args else None

        return _method


_BACKEND = _MockBackend()


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

    def test_repr_includes_data_preview(self) -> None:
        class FakeData:
            def __repr__(self) -> str:
                return "shape: (3, 2)\n| id | name |"

        df = DataFrame(_data=FakeData(), _schema=Users)
        r = repr(df)
        assert r.startswith("DataFrame[Users]\n")
        assert "shape: (3, 2)" in r

    def test_repr_html_delegates_to_data(self) -> None:
        class FakeData:
            def _repr_html_(self) -> str:
                return "<table>rows</table>"

        df = DataFrame(_data=FakeData(), _schema=Users)
        html = df._repr_html_()
        assert html is not None
        assert "<b>DataFrame[Users]</b>" in html
        assert "<table>rows</table>" in html

    def test_repr_html_returns_none_without_data(self) -> None:
        df = DataFrame(_schema=Users)
        assert df._repr_html_() is None

    def test_to_native_returns_inner_data(self) -> None:
        sentinel = object()
        df = DataFrame(_data=sentinel, _schema=Users)
        assert df.to_native() is sentinel

    def test_to_native_returns_none_when_no_data(self) -> None:
        df = DataFrame(_schema=Users)
        assert df.to_native() is None


# ---------------------------------------------------------------------------
# Schema-preserving operations
# ---------------------------------------------------------------------------


class TestSchemaPreservingOps:
    def setup_method(self) -> None:
        self.df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)

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
        self.df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)

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
        df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)
        gb = df.group_by(Users.age)
        assert isinstance(gb, GroupBy)

    def test_group_by_keys_stored(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)
        gb = df.group_by(Users.age, Users.name)
        assert gb._keys == (Users.age, Users.name)

    def test_group_by_schema_stored(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)
        gb = df.group_by(Users.age)
        assert gb._schema is Users

    def test_agg_returns_dataframe(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)
        result = df.group_by(Users.age).agg(Users.id.count().alias("count"))
        assert isinstance(result, DataFrame)
        assert result._schema is None

    def test_agg_multiple_exprs(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)
        result = df.group_by(Users.age).agg(
            Users.id.count().alias("count"),
            Users.name.count().alias("name_count"),
        )
        assert isinstance(result, DataFrame)


# ---------------------------------------------------------------------------
# Conversion: lazy, collect
# ---------------------------------------------------------------------------


class TestConversion:
    def test_lazy_returns_lazyframe(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)
        result = df.lazy()
        assert isinstance(result, LazyFrame)
        assert result._schema is Users

    def test_collect_returns_dataframe(self) -> None:
        lf: LazyFrame[Users] = LazyFrame(_schema=Users, _backend=_BACKEND)
        result = lf.collect()
        assert isinstance(result, DataFrame)
        assert result._schema is Users

    def test_validate_returns_self(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users)
        result = df.validate()
        assert result is df

    def test_to_batches_no_backend_raises(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            list(df.to_batches())


# ---------------------------------------------------------------------------
# LazyFrame operations
# ---------------------------------------------------------------------------


class TestLazyFrame:
    def setup_method(self) -> None:
        self.lf: LazyFrame[Users] = LazyFrame(_schema=Users, _backend=_BACKEND)

    def test_repr_with_schema(self) -> None:
        lf = LazyFrame(_schema=Users)
        assert repr(lf) == "LazyFrame[Users]"

    def test_repr_without_schema(self) -> None:
        lf = LazyFrame()
        assert repr(lf) == "LazyFrame[Any]"

    def test_repr_includes_data_preview(self) -> None:
        class FakeData:
            def __repr__(self) -> str:
                return "naive query plan"

        lf = LazyFrame(_data=FakeData(), _schema=Users)
        r = repr(lf)
        assert r.startswith("LazyFrame[Users]\n")
        assert "naive query plan" in r

    def test_to_native_returns_inner_data(self) -> None:
        sentinel = object()
        lf = LazyFrame(_data=sentinel, _schema=Users)
        assert lf.to_native() is sentinel

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

    def test_head_returns_lazyframe(self) -> None:
        result = self.lf.head(10)
        assert isinstance(result, LazyFrame)
        assert result._schema is Users

    def test_tail_returns_lazyframe(self) -> None:
        result = self.lf.tail(10)
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

    def test_validate_returns_self(self) -> None:
        lf = LazyFrame(_schema=Users)
        result = lf.validate()
        assert result is lf


# ---------------------------------------------------------------------------
# LazyFrame restrictions (no sample)
# ---------------------------------------------------------------------------


class TestLazyFrameRestrictions:
    def test_no_sample(self) -> None:
        assert not hasattr(LazyFrame, "sample")

    def test_has_to_batches(self) -> None:
        assert hasattr(LazyFrame, "to_batches")


# ---------------------------------------------------------------------------
# LazyGroupBy
# ---------------------------------------------------------------------------


class TestLazyGroupBy:
    def test_lazy_groupby_agg_returns_lazyframe(self) -> None:
        lf: LazyFrame[Users] = LazyFrame(_schema=Users, _backend=_BACKEND)
        result = lf.group_by(Users.age).agg(Users.id.count().alias("count"))
        assert isinstance(result, LazyFrame)
        assert result._schema is None

    def test_lazy_groupby_keys_stored(self) -> None:
        lf: LazyFrame[Users] = LazyFrame(_schema=Users, _backend=_BACKEND)
        gb = lf.group_by(Users.age, Users.name)
        assert gb._keys == (Users.age, Users.name)


# ---------------------------------------------------------------------------
# Introspection properties (DataFrame)
# ---------------------------------------------------------------------------


class TestDataFrameIntrospection:
    def setup_method(self) -> None:
        self.df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)

    def test_height_returns_int(self) -> None:
        assert self.df.height == 0
        assert isinstance(self.df.height, int)

    def test_len_returns_int(self) -> None:
        assert len(self.df) == 0
        assert isinstance(len(self.df), int)

    def test_len_equals_height(self) -> None:
        assert len(self.df) == self.df.height

    def test_width_returns_column_count(self) -> None:
        assert self.df.width == 3  # id, name, age

    def test_shape_returns_tuple(self) -> None:
        assert self.df.shape == (0, 3)

    def test_is_empty_returns_bool(self) -> None:
        assert self.df.is_empty() is True
        assert isinstance(self.df.is_empty(), bool)

    def test_width_no_schema_raises_type_error(self) -> None:
        df: DataFrame[Users] = DataFrame(_backend=_BACKEND)  # schema=None
        with pytest.raises(TypeError, match="width is not available"):
            df.width  # noqa: B018

    def test_shape_no_schema_raises_type_error(self) -> None:
        df: DataFrame[Users] = DataFrame(_backend=_BACKEND)
        with pytest.raises(TypeError, match="width is not available"):
            df.shape  # noqa: B018

    def test_height_no_backend_raises(self) -> None:
        df = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            df.height  # noqa: B018

    def test_len_no_backend_raises(self) -> None:
        df = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            len(df)

    def test_shape_no_backend_raises(self) -> None:
        df = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            df.shape  # noqa: B018

    def test_is_empty_no_backend_raises(self) -> None:
        df = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            df.is_empty()


# ---------------------------------------------------------------------------
# Introspection properties (LazyFrame)
# ---------------------------------------------------------------------------


class TestLazyFrameIntrospection:
    def test_width_returns_column_count(self) -> None:
        lf: LazyFrame[Users] = LazyFrame(_schema=Users, _backend=_BACKEND)
        assert lf.width == 3

    def test_width_no_schema_raises_type_error(self) -> None:
        lf: LazyFrame[Users] = LazyFrame(_backend=_BACKEND)
        with pytest.raises(TypeError, match="width is not available"):
            lf.width  # noqa: B018

    def test_height_returns_row_count(self) -> None:
        lf = self.lf if hasattr(self, "lf") else LazyFrame(_schema=Users, _backend=_BACKEND)
        assert isinstance(lf.height, int)

    def test_len_returns_row_count(self) -> None:
        lf = self.lf if hasattr(self, "lf") else LazyFrame(_schema=Users, _backend=_BACKEND)
        assert isinstance(len(lf), int)


# ---------------------------------------------------------------------------
# LazyFrame restrictions (no shape, is_empty, iter_rows_as)
# ---------------------------------------------------------------------------


class TestLazyFrameIntrospectionRestrictions:
    def test_no_shape(self) -> None:
        assert not hasattr(LazyFrame, "shape")

    def test_no_is_empty(self) -> None:
        assert not hasattr(LazyFrame, "is_empty")

    def test_no_iter_rows_as(self) -> None:
        assert not hasattr(LazyFrame, "iter_rows_as")


# ---------------------------------------------------------------------------
# iter_rows_as
# ---------------------------------------------------------------------------


class TestIterRowsAs:
    def test_returns_iterator(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)
        result = df.iter_rows_as(dict)
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_yields_nothing_for_empty(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)
        rows = list(df.iter_rows_as(dict))
        assert rows == []

    def test_no_backend_raises(self) -> None:
        df = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            list(df.iter_rows_as(dict))

    def test_incompatible_row_type_raises(self) -> None:
        """Passing a type that can't accept **kwargs raises TypeError at iteration."""

        class _MockBackendWithRows:
            def row_count(self, source: object) -> int:  # noqa: ANN001
                return 1

            def iter_row_dicts(self, source: object) -> list[dict[str, object]]:  # noqa: ANN001
                return [{"id": 1, "name": "Alice", "age": 25}]

            def __getattr__(self, name: str):  # noqa: ANN204
                def _method(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                    return args[0] if args else None

                return _method

        df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_MockBackendWithRows())
        with pytest.raises(TypeError):
            list(df.iter_rows_as(int))  # int() does not accept **kwargs


# ---------------------------------------------------------------------------
# JoinedDataFrame / JoinedLazyFrame introspection restrictions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _validate_expr_columns edge case
# ---------------------------------------------------------------------------


class TestValidateExprColumns:
    def test_schema_none_returns_immediately(self) -> None:
        from colnade.dataframe import _validate_expr_columns

        # Should not raise even with garbage args when schema is None
        _validate_expr_columns(None, "anything", 42)


from colnade import JoinedDataFrame, JoinedLazyFrame  # noqa: E402

# ---------------------------------------------------------------------------
# LazyFrame repr edge cases
# ---------------------------------------------------------------------------


class TestLazyFrameReprEdgeCases:
    def test_repr_html_delegates_to_data(self) -> None:
        class FakeData:
            def _repr_html_(self) -> str:
                return "<table>lazy plan</table>"

        lf = LazyFrame(_data=FakeData(), _schema=Users)
        html = lf._repr_html_()
        assert html is not None
        assert "<b>LazyFrame[Users]</b>" in html
        assert "<table>lazy plan</table>" in html

    def test_repr_html_returns_none_without_data(self) -> None:
        lf = LazyFrame(_schema=Users)
        assert lf._repr_html_() is None

    def test_repr_html_returns_none_without_repr_html_method(self) -> None:
        lf = LazyFrame(_data="plain string", _schema=Users)
        assert lf._repr_html_() is None


# ---------------------------------------------------------------------------
# JoinedDataFrame repr
# ---------------------------------------------------------------------------


class TestJoinedDataFrameRepr:
    def test_repr_with_data(self) -> None:
        class FakeData:
            def __repr__(self) -> str:
                return "joined data"

        jdf = JoinedDataFrame(
            _data=FakeData(), _schema_left=Users, _schema_right=AgeStats, _backend=_BACKEND
        )
        r = repr(jdf)
        assert "JoinedDataFrame[Users, AgeStats]" in r
        assert "joined data" in r

    def test_repr_without_data(self) -> None:
        jdf = JoinedDataFrame(_schema_left=Users, _schema_right=AgeStats, _backend=_BACKEND)
        assert repr(jdf) == "JoinedDataFrame[Users, AgeStats]"

    def test_repr_html_with_data(self) -> None:
        class FakeData:
            def _repr_html_(self) -> str:
                return "<table>joined</table>"

        jdf = JoinedDataFrame(
            _data=FakeData(), _schema_left=Users, _schema_right=AgeStats, _backend=_BACKEND
        )
        html = jdf._repr_html_()
        assert html is not None
        assert "<b>JoinedDataFrame[Users, AgeStats]</b>" in html

    def test_repr_html_returns_none_without_data(self) -> None:
        jdf = JoinedDataFrame(_schema_left=Users, _schema_right=AgeStats, _backend=_BACKEND)
        assert jdf._repr_html_() is None


# ---------------------------------------------------------------------------
# JoinedLazyFrame repr
# ---------------------------------------------------------------------------


class TestJoinedLazyFrameRepr:
    def test_repr_with_data(self) -> None:
        class FakeData:
            def __repr__(self) -> str:
                return "joined lazy plan"

        jlf = JoinedLazyFrame(
            _data=FakeData(), _schema_left=Users, _schema_right=AgeStats, _backend=_BACKEND
        )
        r = repr(jlf)
        assert "JoinedLazyFrame[Users, AgeStats]" in r
        assert "joined lazy plan" in r

    def test_repr_without_data(self) -> None:
        jlf = JoinedLazyFrame(_schema_left=Users, _schema_right=AgeStats, _backend=_BACKEND)
        assert repr(jlf) == "JoinedLazyFrame[Users, AgeStats]"

    def test_repr_html_with_data(self) -> None:
        class FakeData:
            def _repr_html_(self) -> str:
                return "<table>joined lazy</table>"

        jlf = JoinedLazyFrame(
            _data=FakeData(), _schema_left=Users, _schema_right=AgeStats, _backend=_BACKEND
        )
        html = jlf._repr_html_()
        assert html is not None
        assert "<b>JoinedLazyFrame[Users, AgeStats]</b>" in html

    def test_repr_html_returns_none_without_data(self) -> None:
        jlf = JoinedLazyFrame(_schema_left=Users, _schema_right=AgeStats, _backend=_BACKEND)
        assert jlf._repr_html_() is None


class TestJoinedIntrospectionRestrictions:
    def test_joined_df_no_height(self) -> None:
        assert not hasattr(JoinedDataFrame, "height")

    def test_joined_df_no_width(self) -> None:
        assert not hasattr(JoinedDataFrame, "width")

    def test_joined_df_no_shape(self) -> None:
        assert not hasattr(JoinedDataFrame, "shape")

    def test_joined_df_no_is_empty(self) -> None:
        assert not hasattr(JoinedDataFrame, "is_empty")

    def test_joined_df_no_iter_rows_as(self) -> None:
        assert not hasattr(JoinedDataFrame, "iter_rows_as")

    def test_joined_lf_no_height(self) -> None:
        assert not hasattr(JoinedLazyFrame, "height")

    def test_joined_lf_no_width(self) -> None:
        assert not hasattr(JoinedLazyFrame, "width")

    def test_joined_lf_no_shape(self) -> None:
        assert not hasattr(JoinedLazyFrame, "shape")

    def test_joined_lf_no_is_empty(self) -> None:
        assert not hasattr(JoinedLazyFrame, "is_empty")

    def test_joined_lf_no_iter_rows_as(self) -> None:
        assert not hasattr(JoinedLazyFrame, "iter_rows_as")

    def test_joined_df_no_len(self) -> None:
        jdf = JoinedDataFrame(_backend=_BACKEND)
        with pytest.raises(TypeError):
            len(jdf)

    def test_joined_lf_no_len(self) -> None:
        jlf = JoinedLazyFrame(_backend=_BACKEND)
        with pytest.raises(TypeError):
            len(jlf)


# ---------------------------------------------------------------------------
# No-backend RuntimeError tests
# ---------------------------------------------------------------------------


class TestNoBackendRaisesRuntimeError:
    """Operations on frames without a backend must raise RuntimeError."""

    def test_dataframe_filter(self) -> None:
        df = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            df.filter(Users.age > 18)

    def test_dataframe_sort(self) -> None:
        df = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            df.sort(Users.name)

    def test_dataframe_limit(self) -> None:
        df = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            df.limit(10)

    def test_dataframe_head(self) -> None:
        df = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            df.head()

    def test_dataframe_select(self) -> None:
        df = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            df.select(Users.name)

    def test_dataframe_lazy(self) -> None:
        df = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            df.lazy()

    def test_dataframe_cast_schema(self) -> None:
        df = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            df.cast_schema(AgeStats)

    def test_lazyframe_filter(self) -> None:
        lf = LazyFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            lf.filter(Users.age > 18)

    def test_lazyframe_collect(self) -> None:
        lf = LazyFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            lf.collect()

    def test_groupby_agg(self) -> None:
        df = DataFrame(_schema=Users, _backend=_BACKEND)
        gb = df.group_by(Users.age)
        gb._backend = None  # simulate missing backend
        with pytest.raises(RuntimeError, match="requires a backend"):
            gb.agg(Users.id.count().alias("count"))

    def test_lazy_groupby_agg(self) -> None:
        lf = LazyFrame(_schema=Users, _backend=_BACKEND)
        lgb = lf.group_by(Users.age)
        lgb._backend = None
        with pytest.raises(RuntimeError, match="requires a backend"):
            lgb.agg(Users.id.count().alias("count"))


# ---------------------------------------------------------------------------
# concat()
# ---------------------------------------------------------------------------


class TestConcat:
    def test_concat_two_dataframes(self) -> None:
        df1 = DataFrame(_data="a", _schema=Users, _backend=_BACKEND)
        df2 = DataFrame(_data="b", _schema=Users, _backend=_BACKEND)
        result = concat(df1, df2)
        assert isinstance(result, DataFrame)
        assert result._schema is Users
        assert result._backend is _BACKEND

    def test_concat_three_dataframes(self) -> None:
        df1 = DataFrame(_data="a", _schema=Users, _backend=_BACKEND)
        df2 = DataFrame(_data="b", _schema=Users, _backend=_BACKEND)
        df3 = DataFrame(_data="c", _schema=Users, _backend=_BACKEND)
        result = concat(df1, df2, df3)
        assert isinstance(result, DataFrame)

    def test_concat_lazyframes(self) -> None:
        lf1 = LazyFrame(_data="a", _schema=Users, _backend=_BACKEND)
        lf2 = LazyFrame(_data="b", _schema=Users, _backend=_BACKEND)
        result = concat(lf1, lf2)
        assert isinstance(result, LazyFrame)
        assert result._schema is Users

    def test_concat_fewer_than_two_raises(self) -> None:
        df1 = DataFrame(_data="a", _schema=Users, _backend=_BACKEND)
        with pytest.raises(ValueError, match="at least 2"):
            concat(df1)
        with pytest.raises(ValueError, match="at least 2"):
            concat()

    def test_concat_mismatched_schemas_raises(self) -> None:
        df1 = DataFrame(_data="a", _schema=Users, _backend=_BACKEND)
        df2 = DataFrame(_data="b", _schema=AgeStats, _backend=_BACKEND)
        with pytest.raises(ValueError, match="Schema mismatch"):
            concat(df1, df2)

    def test_concat_mixed_frame_types_raises(self) -> None:
        df = DataFrame(_data="a", _schema=Users, _backend=_BACKEND)
        lf = LazyFrame(_data="b", _schema=Users, _backend=_BACKEND)
        with pytest.raises(TypeError, match="cannot mix"):
            concat(df, lf)  # type: ignore[call-overload]

    def test_concat_no_backend_raises(self) -> None:
        df1 = DataFrame(_data="a", _schema=Users)
        df2 = DataFrame(_data="b", _schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            concat(df1, df2)


# ---------------------------------------------------------------------------
# .item() — scalar extraction
# ---------------------------------------------------------------------------


class TestItem:
    def test_item_no_column(self) -> None:
        df = DataFrame(_data="src", _schema=Users, _backend=_BACKEND)
        result = df.item()
        assert result == ("item_called", "src", None)

    def test_item_with_column(self) -> None:
        df = DataFrame(_data="src", _schema=Users, _backend=_BACKEND)
        result = df.item(Users.name)
        assert result == ("item_called", "src", "name")

    def test_item_on_lazyframe(self) -> None:
        lf = LazyFrame(_data="src", _schema=Users, _backend=_BACKEND)
        result = lf.item()
        assert result == ("item_called", "src", None)

    def test_item_with_column_on_lazyframe(self) -> None:
        lf = LazyFrame(_data="src", _schema=Users, _backend=_BACKEND)
        result = lf.item(Users.age)
        assert result == ("item_called", "src", "age")
