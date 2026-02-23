"""Integration tests for Dask I/O functions."""

from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pytest

import colnade.validation
from colnade import Column, LazyFrame, Schema, SchemaError, UInt64, Utf8
from colnade_dask.adapter import DaskBackend
from colnade_dask.io import (
    scan_csv,
    scan_parquet,
    write_csv,
    write_parquet,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]


class WrongSchema(Schema):
    id: Column[UInt64]
    email: Column[Utf8]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_ddf() -> dd.DataFrame:
    pdf = pd.DataFrame(
        {
            "id": pd.array([1, 2, 3], dtype=pd.UInt64Dtype()),
            "name": pd.array(["Alice", "Bob", "Charlie"], dtype=pd.StringDtype()),
            "age": pd.array([30, 25, 35], dtype=pd.UInt64Dtype()),
        }
    )
    return dd.from_pandas(pdf, npartitions=2)


# ---------------------------------------------------------------------------
# Parquet roundtrip
# ---------------------------------------------------------------------------


class TestParquetRoundtrip:
    def test_write_scan_parquet(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.parquet")
        data = _sample_ddf()
        lf = LazyFrame(_data=data, _schema=Users, _backend=DaskBackend())
        write_parquet(lf, path)

        result = scan_parquet(path, Users)
        assert isinstance(result, LazyFrame)
        assert result._schema is Users
        collected = result.collect()
        computed = collected._data.compute()
        assert computed.shape == (3, 3)
        assert computed["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_scan_parquet_filter_collect(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.parquet")
        _sample_ddf().to_parquet(path)

        result = scan_parquet(path, Users).filter(Users.age > 28).collect()
        assert result._data.compute().shape[0] == 2


# ---------------------------------------------------------------------------
# CSV roundtrip
# ---------------------------------------------------------------------------


class TestCsvRoundtrip:
    def test_write_scan_csv(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.csv")
        data = _sample_ddf()
        lf = LazyFrame(_data=data, _schema=Users, _backend=DaskBackend())
        write_csv(lf, path)

        result = scan_csv(path, Users)
        assert isinstance(result, LazyFrame)
        collected = result.collect()
        assert collected._data.compute().shape == (3, 3)


# ---------------------------------------------------------------------------
# Schema mismatch
# ---------------------------------------------------------------------------


class TestSchemaMismatch:
    def setup_method(self) -> None:
        colnade.validation.set_validation(True)

    def teardown_method(self) -> None:
        colnade.validation._validation_level = None

    def test_scan_parquet_wrong_schema(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.parquet")
        _sample_ddf().to_parquet(path)

        with pytest.raises(SchemaError) as exc_info:
            scan_parquet(path, WrongSchema)
        assert "email" in exc_info.value.missing_columns
