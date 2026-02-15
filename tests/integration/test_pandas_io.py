"""Integration tests for Pandas I/O functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import colnade.validation
from colnade import Column, DataFrame, Schema, SchemaError, UInt64, Utf8
from colnade_pandas.adapter import PandasBackend
from colnade_pandas.io import (
    read_csv,
    read_parquet,
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


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": pd.array([1, 2, 3], dtype=pd.UInt64Dtype()),
            "name": pd.array(["Alice", "Bob", "Charlie"], dtype=pd.StringDtype()),
            "age": pd.array([30, 25, 35], dtype=pd.UInt64Dtype()),
        }
    )


# ---------------------------------------------------------------------------
# Parquet roundtrip
# ---------------------------------------------------------------------------


class TestParquetRoundtrip:
    def test_write_read_parquet(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.parquet")
        data = _sample_data()
        df = DataFrame(_data=data, _schema=Users, _backend=PandasBackend())
        write_parquet(df, path)

        result = read_parquet(path, Users)
        assert isinstance(result, DataFrame)
        assert result._schema is Users
        assert result._data.shape == (3, 3)
        assert result._data["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_read_parquet_filter(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.parquet")
        _sample_data().to_parquet(path)

        result = read_parquet(path, Users).filter(Users.age > 28)
        assert result._data.shape[0] == 2


# ---------------------------------------------------------------------------
# CSV roundtrip
# ---------------------------------------------------------------------------


class TestCsvRoundtrip:
    def test_write_read_csv(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.csv")
        data = _sample_data()
        df = DataFrame(_data=data, _schema=Users, _backend=PandasBackend())
        write_csv(df, path)

        result = read_csv(path, Users)
        assert isinstance(result, DataFrame)
        assert result._schema is Users
        assert result._data.shape == (3, 3)
        assert result._data["name"].tolist() == ["Alice", "Bob", "Charlie"]


# ---------------------------------------------------------------------------
# Schema mismatch
# ---------------------------------------------------------------------------


class TestSchemaMismatch:
    def setup_method(self) -> None:
        colnade.validation.set_validation(True)

    def teardown_method(self) -> None:
        colnade.validation._validation_enabled = None

    def test_read_parquet_wrong_schema(self, tmp_path: Path) -> None:
        path = str(tmp_path / "users.parquet")
        _sample_data().to_parquet(path)

        with pytest.raises(SchemaError) as exc_info:
            read_parquet(path, WrongSchema)
        assert "email" in exc_info.value.missing_columns
