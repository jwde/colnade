"""Read/write operations for Pandas backend."""

from __future__ import annotations

from typing import Any, TypeVar

import pandas as pd

from colnade import DataFrame, Schema
from colnade.validation import get_validation_level, is_validation_enabled
from colnade_pandas.adapter import PandasBackend
from colnade_pandas.conversion import map_colnade_dtype

S = TypeVar("S", bound=Schema)


def _build_pandas_schema(schema: type[S]) -> dict[str, Any]:
    """Build a Pandas dtype dict from a Colnade schema."""
    return {name: map_colnade_dtype(col.dtype) for name, col in schema._columns.items()}


def read_parquet(path: str, schema: type[S], **kwargs: Any) -> DataFrame[S]:
    """Read a Parquet file into a typed DataFrame."""
    backend = PandasBackend()
    data = pd.read_parquet(path, **kwargs)
    if is_validation_enabled():
        backend.validate_schema(data, schema)
        if get_validation_level() == "full":
            backend.validate_field_constraints(data, schema)
    return DataFrame(_data=data, _schema=schema, _backend=backend)


def read_csv(path: str, schema: type[S], **kwargs: Any) -> DataFrame[S]:
    """Read a CSV file into a typed DataFrame.

    Applies the schema's dtype mapping to ensure correct column types.
    """
    backend = PandasBackend()
    pd_schema = _build_pandas_schema(schema)
    data = pd.read_csv(path, dtype=pd_schema, **kwargs)
    if is_validation_enabled():
        backend.validate_schema(data, schema)
        if get_validation_level() == "full":
            backend.validate_field_constraints(data, schema)
    return DataFrame(_data=data, _schema=schema, _backend=backend)


def write_parquet(df: DataFrame[Any], path: str, **kwargs: Any) -> None:
    """Write a DataFrame to a Parquet file."""
    df._data.to_parquet(path, **kwargs)


def write_csv(df: DataFrame[Any], path: str, **kwargs: Any) -> None:
    """Write a DataFrame to a CSV file."""
    df._data.to_csv(path, index=False, **kwargs)
