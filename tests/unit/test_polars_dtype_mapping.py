"""Unit tests for dtype mapping between Colnade and Polars."""

from __future__ import annotations

import polars as pl
import pytest

from colnade import (
    Binary,
    Bool,
    Column,
    Date,
    Datetime,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Schema,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
)
from colnade_polars.conversion import (
    map_colnade_dtype,
    map_polars_dtype,
)

# ---------------------------------------------------------------------------
# Concrete type mapping
# ---------------------------------------------------------------------------


class TestConcreteTypes:
    @pytest.mark.parametrize(
        ("colnade_type", "expected_polars"),
        [
            (Bool, pl.Boolean()),
            (UInt8, pl.UInt8()),
            (UInt16, pl.UInt16()),
            (UInt32, pl.UInt32()),
            (UInt64, pl.UInt64()),
            (Int8, pl.Int8()),
            (Int16, pl.Int16()),
            (Int32, pl.Int32()),
            (Int64, pl.Int64()),
            (Float32, pl.Float32()),
            (Float64, pl.Float64()),
            (Utf8, pl.String()),
            (Binary, pl.Binary()),
            (Date, pl.Date()),
            (Time, pl.Time()),
            (Datetime, pl.Datetime()),
            (Duration, pl.Duration()),
        ],
    )
    def test_colnade_to_polars(self, colnade_type: type, expected_polars: pl.DataType) -> None:
        assert map_colnade_dtype(colnade_type) == expected_polars

    @pytest.mark.parametrize(
        ("polars_dtype", "expected_colnade"),
        [
            (pl.Boolean(), Bool),
            (pl.UInt8(), UInt8),
            (pl.UInt16(), UInt16),
            (pl.UInt32(), UInt32),
            (pl.UInt64(), UInt64),
            (pl.Int8(), Int8),
            (pl.Int16(), Int16),
            (pl.Int32(), Int32),
            (pl.Int64(), Int64),
            (pl.Float32(), Float32),
            (pl.Float64(), Float64),
            (pl.String(), Utf8),
            (pl.Binary(), Binary),
            (pl.Date(), Date),
            (pl.Time(), Time),
            (pl.Datetime(), Datetime),
            (pl.Duration(), Duration),
        ],
    )
    def test_polars_to_colnade(self, polars_dtype: pl.DataType, expected_colnade: type) -> None:
        assert map_polars_dtype(polars_dtype) == expected_colnade


# ---------------------------------------------------------------------------
# Nullable types
# ---------------------------------------------------------------------------


class TestNullableTypes:
    def test_nullable_uint64(self) -> None:
        # UInt64 | None should map to UInt64 (Polars handles nulls natively)
        import typing

        nullable = typing.Union[UInt64, None]  # noqa: UP007
        result = map_colnade_dtype(nullable)
        assert result == pl.UInt64()

    def test_nullable_utf8(self) -> None:
        import typing

        nullable = typing.Union[Utf8, None]  # noqa: UP007
        result = map_colnade_dtype(nullable)
        assert result == pl.String()


# ---------------------------------------------------------------------------
# Struct[S] mapping
# ---------------------------------------------------------------------------


class TestStructMapping:
    def test_struct_schema(self) -> None:
        class Addr(Schema):
            street: Column[Utf8]
            city: Column[Utf8]

        result = map_colnade_dtype(Struct[Addr])
        assert isinstance(result, pl.Struct)
        assert result == pl.Struct([pl.Field("street", pl.String()), pl.Field("city", pl.String())])


# ---------------------------------------------------------------------------
# List[T] mapping
# ---------------------------------------------------------------------------


class TestListMapping:
    def test_list_of_int(self) -> None:
        result = map_colnade_dtype(List[Int64])
        assert result == pl.List(pl.Int64())

    def test_list_of_string(self) -> None:
        result = map_colnade_dtype(List[Utf8])
        assert result == pl.List(pl.String())


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unsupported_colnade_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported Colnade dtype"):
            map_colnade_dtype(object)

    def test_unsupported_polars_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported Polars dtype"):
            map_polars_dtype(pl.Null())
