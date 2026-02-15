"""Integration tests for runtime schema validation with Polars backend."""

from __future__ import annotations

import polars as pl
import pytest

import colnade.validation
from colnade import Column, DataFrame, LazyFrame, Schema, SchemaError, UInt64, Utf8
from colnade.validation import set_validation
from colnade_polars.adapter import PolarsBackend

# ---------------------------------------------------------------------------
# Test fixture schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]


class UsersNullable(Schema):
    id: Column[UInt64]
    name: Column[Utf8 | None]
    age: Column[UInt64 | None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(data: dict[str, list], schema: type = Users) -> DataFrame:
    from colnade_polars.conversion import map_colnade_dtype

    backend = PolarsBackend()
    pl_df = pl.DataFrame(data).cast(
        {name: map_colnade_dtype(col.dtype) for name, col in schema._columns.items()},
        strict=False,
    )
    return DataFrame(_data=pl_df, _schema=schema, _backend=backend)


# ---------------------------------------------------------------------------
# Explicit validate() — always runs
# ---------------------------------------------------------------------------


class TestExplicitValidate:
    def test_valid_dataframe_passes(self) -> None:
        df = _make_df({"id": [1, 2], "name": ["Alice", "Bob"], "age": [30, 25]})
        result = df.validate()
        assert result is df

    def test_missing_column_raises(self) -> None:
        backend = PolarsBackend()
        data = pl.DataFrame({"id": pl.Series([1], dtype=pl.UInt64)})
        df = DataFrame(_data=data, _schema=Users, _backend=backend)
        with pytest.raises(SchemaError, match="Missing columns"):
            df.validate()

    def test_type_mismatch_raises(self) -> None:
        backend = PolarsBackend()
        data = pl.DataFrame(
            {
                "id": pl.Series([1], dtype=pl.UInt64),
                "name": pl.Series(["Alice"], dtype=pl.Utf8),
                "age": pl.Series([30], dtype=pl.Int32),  # Wrong type
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=backend)
        with pytest.raises(SchemaError, match="Type mismatch"):
            df.validate()

    def test_null_in_non_nullable_raises(self) -> None:
        backend = PolarsBackend()
        data = pl.DataFrame(
            {
                "id": pl.Series([1, 2], dtype=pl.UInt64),
                "name": pl.Series(["Alice", None], dtype=pl.Utf8),
                "age": pl.Series([30, 25], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=Users, _backend=backend)
        with pytest.raises(SchemaError, match="Null violation"):
            df.validate()

    def test_null_in_nullable_column_ok(self) -> None:
        backend = PolarsBackend()
        data = pl.DataFrame(
            {
                "id": pl.Series([1, 2], dtype=pl.UInt64),
                "name": pl.Series(["Alice", None], dtype=pl.Utf8),
                "age": pl.Series([30, None], dtype=pl.UInt64),
            }
        )
        df = DataFrame(_data=data, _schema=UsersNullable, _backend=backend)
        result = df.validate()
        assert result is df


# ---------------------------------------------------------------------------
# LazyFrame.validate()
# ---------------------------------------------------------------------------


class TestLazyFrameValidate:
    def test_lazyframe_validate_checks_schema(self) -> None:
        backend = PolarsBackend()
        data = pl.DataFrame({"id": pl.Series([1], dtype=pl.UInt64)}).lazy()
        lf = LazyFrame(_data=data, _schema=Users, _backend=backend)
        with pytest.raises(SchemaError, match="Missing columns"):
            lf.validate()

    def test_lazyframe_validate_passes(self) -> None:
        backend = PolarsBackend()
        data = pl.DataFrame(
            {
                "id": pl.Series([1], dtype=pl.UInt64),
                "name": pl.Series(["Alice"], dtype=pl.Utf8),
                "age": pl.Series([30], dtype=pl.UInt64),
            }
        ).lazy()
        lf = LazyFrame(_data=data, _schema=Users, _backend=backend)
        result = lf.validate()
        assert result is lf


# ---------------------------------------------------------------------------
# Validation toggle — auto-validation at boundaries
# ---------------------------------------------------------------------------


class TestValidationToggle:
    def setup_method(self) -> None:
        colnade.validation._validation_level = None

    def teardown_method(self) -> None:
        colnade.validation._validation_level = None

    def test_read_parquet_no_validation_by_default(self, tmp_path) -> None:
        """With validation off, mismatched parquet loads without error."""
        path = tmp_path / "data.parquet"
        # Write with wrong types
        pl.DataFrame(
            {
                "id": pl.Series([1], dtype=pl.Int32),
                "name": pl.Series(["Alice"], dtype=pl.Utf8),
                "age": pl.Series([30], dtype=pl.Int32),
            }
        ).write_parquet(str(path))

        from colnade_polars.io import read_parquet

        # Should NOT raise — validation is off by default
        df = read_parquet(str(path), Users)
        assert df._schema is Users

    def test_read_parquet_validates_when_enabled(self, tmp_path) -> None:
        """With validation on, mismatched parquet raises SchemaError."""
        path = tmp_path / "data.parquet"
        pl.DataFrame(
            {
                "id": pl.Series([1], dtype=pl.Int32),
                "name": pl.Series(["Alice"], dtype=pl.Utf8),
                "age": pl.Series([30], dtype=pl.Int32),
            }
        ).write_parquet(str(path))

        from colnade_polars.io import read_parquet

        set_validation(True)
        with pytest.raises(SchemaError):
            read_parquet(str(path), Users)

    def test_scan_parquet_validates_when_enabled(self, tmp_path) -> None:
        path = tmp_path / "data.parquet"
        pl.DataFrame(
            {
                "id": pl.Series([1], dtype=pl.Int32),
                "name": pl.Series(["Alice"], dtype=pl.Utf8),
                "age": pl.Series([30], dtype=pl.Int32),
            }
        ).write_parquet(str(path))

        from colnade_polars.io import scan_parquet

        set_validation(True)
        with pytest.raises(SchemaError):
            scan_parquet(str(path), Users)
