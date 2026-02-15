"""Integration tests for Polars I/O functions."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import colnade.validation
from colnade import Column, DataFrame, LazyFrame, Schema, SchemaError, UInt64, Utf8
from colnade_polars.io import (
    read_csv,
    read_parquet,
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


def _sample_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "name": ["Alice", "Bob", "Charlie"],
            "age": pl.Series([30, 25, 35], dtype=pl.UInt64),
        }
    )


# ---------------------------------------------------------------------------
# Parquet roundtrip
# ---------------------------------------------------------------------------


class TestParquetRoundtrip:
    def test_write_read_parquet(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.parquet")
        data = _sample_data()
        from colnade_polars.adapter import PolarsBackend

        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        write_parquet(df, path)

        result = read_parquet(path, Users)
        assert isinstance(result, DataFrame)
        assert result._schema is Users
        assert result._data.shape == (3, 3)
        assert result._data["name"].to_list() == ["Alice", "Bob", "Charlie"]

    def test_scan_parquet(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.parquet")
        _sample_data().write_parquet(path)

        result = scan_parquet(path, Users)
        assert isinstance(result, LazyFrame)
        assert result._schema is Users
        collected = result.collect()
        assert collected._data.shape == (3, 3)

    def test_scan_parquet_filter_collect(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.parquet")
        _sample_data().write_parquet(path)

        result = scan_parquet(path, Users).filter(Users.age > 28).collect()
        assert result._data.shape[0] == 2


# ---------------------------------------------------------------------------
# CSV roundtrip
# ---------------------------------------------------------------------------


class TestCsvRoundtrip:
    def test_write_read_csv(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.csv")
        data = _sample_data()
        from colnade_polars.adapter import PolarsBackend

        df = DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())
        write_csv(df, path)

        result = read_csv(path, Users)
        assert isinstance(result, DataFrame)
        assert result._schema is Users
        assert result._data.shape == (3, 3)
        assert result._data["name"].to_list() == ["Alice", "Bob", "Charlie"]

    def test_scan_csv(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.csv")
        _sample_data().write_csv(path)

        result = scan_csv(path, Users)
        assert isinstance(result, LazyFrame)
        collected = result.collect()
        assert collected._data.shape == (3, 3)


# ---------------------------------------------------------------------------
# Schema mismatch
# ---------------------------------------------------------------------------


class TestSchemaMismatch:
    def setup_method(self) -> None:
        colnade.validation.set_validation(True)

    def teardown_method(self) -> None:
        colnade.validation._validation_level = None

    def test_read_parquet_wrong_schema(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.parquet")
        _sample_data().write_parquet(path)

        with pytest.raises(SchemaError) as exc_info:
            read_parquet(path, WrongSchema)
        assert "email" in exc_info.value.missing_columns
