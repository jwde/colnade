"""Dtype mapping between Colnade and Pandas types."""

from __future__ import annotations

import typing
from typing import Any

import pandas as pd
import pyarrow as pa

from colnade import dtypes

# ---------------------------------------------------------------------------
# Colnade → Pandas mapping (uses nullable extension types)
# ---------------------------------------------------------------------------

COLNADE_TO_PANDAS: dict[type, Any] = {
    dtypes.Bool: pd.BooleanDtype(),
    dtypes.UInt8: pd.UInt8Dtype(),
    dtypes.UInt16: pd.UInt16Dtype(),
    dtypes.UInt32: pd.UInt32Dtype(),
    dtypes.UInt64: pd.UInt64Dtype(),
    dtypes.Int8: pd.Int8Dtype(),
    dtypes.Int16: pd.Int16Dtype(),
    dtypes.Int32: pd.Int32Dtype(),
    dtypes.Int64: pd.Int64Dtype(),
    dtypes.Float32: pd.Float32Dtype(),
    dtypes.Float64: pd.Float64Dtype(),
    dtypes.Utf8: pd.StringDtype(),
    dtypes.Binary: object,
    dtypes.Date: pd.ArrowDtype(pa.date32()),
    dtypes.Time: pd.ArrowDtype(pa.time64("us")),
    dtypes.Datetime: pd.ArrowDtype(pa.timestamp("us")),
    dtypes.Duration: pd.ArrowDtype(pa.duration("us")),
}

# ---------------------------------------------------------------------------
# Pandas → Colnade mapping
# ---------------------------------------------------------------------------

PANDAS_TO_COLNADE: dict[Any, type] = {
    pd.BooleanDtype(): dtypes.Bool,
    pd.UInt8Dtype(): dtypes.UInt8,
    pd.UInt16Dtype(): dtypes.UInt16,
    pd.UInt32Dtype(): dtypes.UInt32,
    pd.UInt64Dtype(): dtypes.UInt64,
    pd.Int8Dtype(): dtypes.Int8,
    pd.Int16Dtype(): dtypes.Int16,
    pd.Int32Dtype(): dtypes.Int32,
    pd.Int64Dtype(): dtypes.Int64,
    pd.Float32Dtype(): dtypes.Float32,
    pd.Float64Dtype(): dtypes.Float64,
    pd.StringDtype(): dtypes.Utf8,
    pd.ArrowDtype(pa.date32()): dtypes.Date,
    pd.ArrowDtype(pa.time64("us")): dtypes.Time,
    pd.ArrowDtype(pa.timestamp("us")): dtypes.Datetime,
    pd.ArrowDtype(pa.duration("us")): dtypes.Duration,
}


def map_colnade_dtype(colnade_type: Any) -> Any:
    """Map a Colnade dtype annotation to a Pandas dtype.

    Handles:
    - Concrete types (UInt64 → pd.UInt64Dtype())
    - Nullable unions (UInt64 | None → pd.UInt64Dtype()) — nullable extension types
    - List[T] / Struct[S] → object
    """
    import types as _types

    origin = typing.get_origin(colnade_type)
    if origin is typing.Union or isinstance(colnade_type, _types.UnionType):
        args = [a for a in typing.get_args(colnade_type) if a is not type(None)]
        if len(args) == 1:
            return map_colnade_dtype(args[0])

    # Struct[S] and List[T] → object
    if origin is dtypes.Struct or origin is dtypes.List:
        return object

    if colnade_type in COLNADE_TO_PANDAS:
        return COLNADE_TO_PANDAS[colnade_type]

    msg = f"Unsupported Colnade dtype: {colnade_type}"
    raise TypeError(msg)


def map_pandas_dtype(pd_dtype: Any) -> type:
    """Map a Pandas dtype to a Colnade dtype class."""
    if pd_dtype in PANDAS_TO_COLNADE:
        return PANDAS_TO_COLNADE[pd_dtype]
    # Handle object dtype (used for Binary, List, Struct)
    if pd_dtype is object:
        return dtypes.Binary
    msg = f"Unsupported Pandas dtype: {pd_dtype}"
    raise TypeError(msg)
