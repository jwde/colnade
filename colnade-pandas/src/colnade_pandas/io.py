"""Read/write operations for Pandas backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import pandas as pd

from colnade import DataFrame, Row, Schema
from colnade.dataframe import rows_to_dict
from colnade.validation import ValidationLevel, get_validation_level, is_validation_enabled
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
        if get_validation_level() is ValidationLevel.FULL:
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
        if get_validation_level() is ValidationLevel.FULL:
            backend.validate_field_constraints(data, schema)
    return DataFrame(_data=data, _schema=schema, _backend=backend)


def from_dict(
    schema: type[S],
    data: dict[str, Sequence[Any]],
) -> DataFrame[S]:
    """Create a typed DataFrame from a columnar dict.

    The schema drives dtype coercion — plain Python values (``[1, 2, 3]``)
    are cast to the correct native types (e.g. ``UInt64``).
    """
    backend = PandasBackend()
    return DataFrame.from_dict(data, schema, backend)


def from_rows(
    schema: type[S],
    rows: Sequence[Row[S]],
) -> DataFrame[S]:
    """Create a typed DataFrame from ``Row[S]`` instances.

    The type checker verifies that rows match the schema — passing
    ``Orders.Row`` where ``Users.Row`` is expected is a static error.
    """
    data = rows_to_dict(rows, schema)
    return from_dict(schema, data)


def write_parquet(df: DataFrame[Any], path: str, **kwargs: Any) -> None:
    """Write a DataFrame to a Parquet file."""
    df._data.to_parquet(path, **kwargs)


def write_csv(df: DataFrame[Any], path: str, **kwargs: Any) -> None:
    """Write a DataFrame to a CSV file."""
    df._data.to_csv(path, index=False, **kwargs)
