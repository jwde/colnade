"""Read/write operations for Polars backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import polars as pl

from colnade import DataFrame, LazyFrame, Schema
from colnade.dataframe import rows_to_dict
from colnade.validation import ValidationLevel, get_validation_level, is_validation_enabled
from colnade_polars.adapter import PolarsBackend
from colnade_polars.conversion import map_colnade_dtype

S = TypeVar("S", bound=Schema)


def _build_polars_schema(schema: type[S]) -> dict[str, pl.DataType]:
    """Build a Polars schema dict from a Colnade schema."""
    return {name: map_colnade_dtype(col.dtype) for name, col in schema._columns.items()}


def read_parquet(path: str, schema: type[S]) -> DataFrame[S]:
    """Read a Parquet file into a typed DataFrame."""
    backend = PolarsBackend()
    data = pl.read_parquet(path)
    if is_validation_enabled():
        backend.validate_schema(data, schema)
        if get_validation_level() is ValidationLevel.FULL:
            backend.validate_field_constraints(data, schema)
    return DataFrame(_data=data, _schema=schema, _backend=backend)


def scan_parquet(path: str, schema: type[S]) -> LazyFrame[S]:
    """Lazily scan a Parquet file into a typed LazyFrame."""
    backend = PolarsBackend()
    data = pl.scan_parquet(path)
    if is_validation_enabled():
        backend.validate_schema(data, schema)
        if get_validation_level() is ValidationLevel.FULL:
            backend.validate_field_constraints(data, schema)
    return LazyFrame(_data=data, _schema=schema, _backend=backend)


def read_csv(path: str, schema: type[S], **kwargs: Any) -> DataFrame[S]:
    """Read a CSV file into a typed DataFrame."""
    backend = PolarsBackend()
    pl_schema = _build_polars_schema(schema)
    data = pl.read_csv(path, schema=pl_schema, **kwargs)
    return DataFrame(_data=data, _schema=schema, _backend=backend)


def scan_csv(path: str, schema: type[S], **kwargs: Any) -> LazyFrame[S]:
    """Lazily scan a CSV file into a typed LazyFrame."""
    backend = PolarsBackend()
    pl_schema = _build_polars_schema(schema)
    data = pl.scan_csv(path, schema=pl_schema, **kwargs)
    return LazyFrame(_data=data, _schema=schema, _backend=backend)


def from_dict(
    schema: type[S],
    data: dict[str, Sequence[Any]],
) -> DataFrame[S]:
    """Create a typed DataFrame from a columnar dict.

    The schema drives dtype coercion — plain Python values (``[1, 2, 3]``)
    are cast to the correct native types (e.g. ``UInt64``).
    """
    backend = PolarsBackend()
    return DataFrame.from_dict(data, schema, backend)


def from_rows(
    schema: type[S],
    rows: Sequence[Any],
) -> DataFrame[S]:
    """Create a typed DataFrame from row objects.

    Accepts ``Schema.Row`` instances, plain dicts, or any object with
    attributes matching the schema's column names.

    Example::

        df = from_rows(Users, [
            Users.Row(id=1, name="Alice", age=30, score=85.0),
            Users.Row(id=2, name="Bob", age=25, score=92.5),
        ])
    """
    data = rows_to_dict(rows, schema)
    return from_dict(schema, data)


def write_parquet(df: DataFrame[Any], path: str) -> None:
    """Write a DataFrame to a Parquet file."""
    df._data.write_parquet(path)


def write_csv(df: DataFrame[Any], path: str, **kwargs: Any) -> None:
    """Write a DataFrame to a CSV file."""
    df._data.write_csv(path, **kwargs)
