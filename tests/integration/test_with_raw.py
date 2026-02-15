"""Integration tests for DataFrame.with_raw() and LazyFrame.with_raw()."""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pytest

import colnade.validation
from colnade import Column, DataFrame, LazyFrame, Schema, SchemaError, UInt64, Utf8
from colnade_dask.adapter import DaskBackend
from colnade_pandas.adapter import PandasBackend
from colnade_polars.adapter import PolarsBackend


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _polars_df() -> DataFrame[Users]:
    backend = PolarsBackend()
    data = pl.DataFrame(
        {
            "id": pl.Series([1, 2], dtype=pl.UInt64),
            "name": pl.Series(["Alice", "Bob"], dtype=pl.Utf8),
            "age": pl.Series([30, 25], dtype=pl.UInt64),
        }
    )
    return DataFrame(_data=data, _schema=Users, _backend=backend)


def _pandas_df() -> DataFrame[Users]:
    backend = PandasBackend()
    data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "age": [30, 25]})
    data = data.astype({"id": "uint64", "age": "uint64", "name": "string"})
    return DataFrame(_data=data, _schema=Users, _backend=backend)


def _dask_df() -> DataFrame[Users]:
    backend = DaskBackend()
    data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "age": [30, 25]})
    data = data.astype({"id": "uint64", "age": "uint64", "name": "string"})
    ddf = dd.from_pandas(data, npartitions=1)
    return DataFrame(_data=ddf, _schema=Users, _backend=backend)


# ---------------------------------------------------------------------------
# DataFrame.with_raw — Polars
# ---------------------------------------------------------------------------


class TestWithRawPolars:
    def test_identity(self) -> None:
        df = _polars_df()
        result = df.with_raw(lambda raw: raw)
        assert result._schema is Users
        assert result._data.shape == (2, 3)

    def test_modify_column(self) -> None:
        df = _polars_df()
        result = df.with_raw(lambda raw: raw.with_columns(pl.col("age") * 2))
        assert result._data["age"].to_list() == [60, 50]
        assert result._schema is Users

    def test_schema_break_with_validation_raises(self) -> None:
        df = _polars_df()
        colnade.validation.set_validation("structural")
        try:
            with pytest.raises(SchemaError, match="Missing columns"):
                df.with_raw(lambda raw: raw.drop("age"))
        finally:
            colnade.validation._validation_level = None

    def test_schema_break_without_validation_ok(self) -> None:
        df = _polars_df()
        colnade.validation.set_validation("off")
        try:
            result = df.with_raw(lambda raw: raw.drop("age"))
            assert result._schema is Users
        finally:
            colnade.validation._validation_level = None


# ---------------------------------------------------------------------------
# DataFrame.with_raw — Pandas
# ---------------------------------------------------------------------------


class TestWithRawPandas:
    def test_identity(self) -> None:
        df = _pandas_df()
        result = df.with_raw(lambda raw: raw)
        assert result._schema is Users
        assert result._data.shape == (2, 3)

    def test_modify_column(self) -> None:
        df = _pandas_df()
        result = df.with_raw(lambda raw: raw.assign(age=raw["age"] * 2))
        assert list(result._data["age"]) == [60, 50]

    def test_schema_break_with_validation_raises(self) -> None:
        df = _pandas_df()
        colnade.validation.set_validation("structural")
        try:
            with pytest.raises(SchemaError, match="Missing columns"):
                df.with_raw(lambda raw: raw.drop(columns=["age"]))
        finally:
            colnade.validation._validation_level = None


# ---------------------------------------------------------------------------
# DataFrame.with_raw — Dask
# ---------------------------------------------------------------------------


class TestWithRawDask:
    def test_identity(self) -> None:
        df = _dask_df()
        result = df.with_raw(lambda raw: raw)
        assert result._schema is Users
        assert result._data.compute().shape == (2, 3)

    def test_modify_column(self) -> None:
        df = _dask_df()
        result = df.with_raw(lambda raw: raw.assign(age=raw["age"] * 2))
        assert list(result._data.compute()["age"]) == [60, 50]

    def test_schema_break_with_validation_raises(self) -> None:
        df = _dask_df()
        colnade.validation.set_validation("structural")
        try:
            with pytest.raises(SchemaError, match="Missing columns"):
                df.with_raw(lambda raw: raw.drop(columns=["age"]))
        finally:
            colnade.validation._validation_level = None


# ---------------------------------------------------------------------------
# LazyFrame.with_raw — Polars
# ---------------------------------------------------------------------------


class TestWithRawLazyPolars:
    def test_identity(self) -> None:
        df = _polars_df()
        lf = df.lazy()
        result = lf.with_raw(lambda raw: raw)
        assert isinstance(result, LazyFrame)
        assert result._schema is Users

    def test_modify_column(self) -> None:
        df = _polars_df()
        lf = df.lazy()
        result = lf.with_raw(lambda raw: raw.with_columns(pl.col("age") * 2))
        collected = result.collect()
        assert collected._data["age"].to_list() == [60, 50]


# ---------------------------------------------------------------------------
# Not available on JoinedDataFrame
# ---------------------------------------------------------------------------


class TestWithRawNotOnJoined:
    def test_joined_dataframe_no_with_raw(self) -> None:
        from colnade import JoinedDataFrame

        assert not hasattr(JoinedDataFrame, "with_raw")

    def test_joined_lazyframe_no_with_raw(self) -> None:
        from colnade import JoinedLazyFrame

        assert not hasattr(JoinedLazyFrame, "with_raw")
