"""Dtype mapping between Colnade and Polars types."""

from __future__ import annotations

import typing
from typing import Any

import polars as pl

from colnade import dtypes

# ---------------------------------------------------------------------------
# Colnade → Polars mapping
# ---------------------------------------------------------------------------

COLNADE_TO_POLARS: dict[type, pl.DataType] = {
    dtypes.Bool: pl.Boolean(),
    dtypes.UInt8: pl.UInt8(),
    dtypes.UInt16: pl.UInt16(),
    dtypes.UInt32: pl.UInt32(),
    dtypes.UInt64: pl.UInt64(),
    dtypes.Int8: pl.Int8(),
    dtypes.Int16: pl.Int16(),
    dtypes.Int32: pl.Int32(),
    dtypes.Int64: pl.Int64(),
    dtypes.Float32: pl.Float32(),
    dtypes.Float64: pl.Float64(),
    dtypes.Utf8: pl.String(),
    dtypes.Binary: pl.Binary(),
    dtypes.Date: pl.Date(),
    dtypes.Time: pl.Time(),
    dtypes.Datetime: pl.Datetime(),
    dtypes.Duration: pl.Duration(),
}

# ---------------------------------------------------------------------------
# Polars → Colnade mapping (keyed by Polars DataType class, not instance)
# ---------------------------------------------------------------------------

POLARS_TO_COLNADE: dict[type[pl.DataType], type] = {
    pl.Boolean: dtypes.Bool,
    pl.UInt8: dtypes.UInt8,
    pl.UInt16: dtypes.UInt16,
    pl.UInt32: dtypes.UInt32,
    pl.UInt64: dtypes.UInt64,
    pl.Int8: dtypes.Int8,
    pl.Int16: dtypes.Int16,
    pl.Int32: dtypes.Int32,
    pl.Int64: dtypes.Int64,
    pl.Float32: dtypes.Float32,
    pl.Float64: dtypes.Float64,
    pl.String: dtypes.Utf8,
    pl.Utf8: dtypes.Utf8,
    pl.Binary: dtypes.Binary,
    pl.Date: dtypes.Date,
    pl.Time: dtypes.Time,
    pl.Datetime: dtypes.Datetime,
    pl.Duration: dtypes.Duration,
}


def map_colnade_dtype(colnade_type: Any) -> pl.DataType:
    """Map a Colnade dtype annotation to a Polars DataType.

    Handles:
    - Concrete types (UInt64 → pl.UInt64)
    - Nullable unions (UInt64 | None → pl.UInt64) — Polars handles nulls natively
    - Struct[S] → pl.Struct with recursively mapped fields
    - List[T] → pl.List with recursively mapped element type
    """
    # Handle nullable unions: T | None → strip None, map T
    origin = typing.get_origin(colnade_type)
    if origin is typing.Union:
        args = [a for a in typing.get_args(colnade_type) if a is not type(None)]
        if len(args) == 1:
            return map_colnade_dtype(args[0])

    # Handle Struct[S]
    if origin is dtypes.Struct:
        schema_args = typing.get_args(colnade_type)
        if schema_args:
            schema_cls = schema_args[0]
            fields = []
            for col_name, col in schema_cls._columns.items():
                pl_dtype = map_colnade_dtype(col.dtype)
                fields.append(pl.Field(col_name, pl_dtype))
            return pl.Struct(fields)

    # Handle List[T]
    if origin is dtypes.List:
        elem_args = typing.get_args(colnade_type)
        if elem_args:
            return pl.List(map_colnade_dtype(elem_args[0]))

    # Concrete type lookup
    if colnade_type in COLNADE_TO_POLARS:
        return COLNADE_TO_POLARS[colnade_type]

    msg = f"Unsupported Colnade dtype: {colnade_type}"
    raise TypeError(msg)


def map_polars_dtype(pl_dtype: pl.DataType) -> type:
    """Map a Polars DataType instance to a Colnade dtype class."""
    dtype_cls = type(pl_dtype)
    if dtype_cls in POLARS_TO_COLNADE:
        return POLARS_TO_COLNADE[dtype_cls]
    msg = f"Unsupported Polars dtype: {pl_dtype}"
    raise TypeError(msg)
