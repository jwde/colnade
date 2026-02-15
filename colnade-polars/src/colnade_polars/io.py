"""Read/write operations for Polars backend."""

from __future__ import annotations

from typing import Any, TypeVar

import polars as pl

from colnade import DataFrame, LazyFrame, Schema
from colnade.validation import get_validation_level, is_validation_enabled
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
        if get_validation_level() == "full":
            backend.validate_field_constraints(data, schema)
    return DataFrame(_data=data, _schema=schema, _backend=backend)


def scan_parquet(path: str, schema: type[S]) -> LazyFrame[S]:
    """Lazily scan a Parquet file into a typed LazyFrame."""
    backend = PolarsBackend()
    data = pl.scan_parquet(path)
    if is_validation_enabled():
        backend.validate_schema(data, schema)
        if get_validation_level() == "full":
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


def write_parquet(df: DataFrame[Any], path: str) -> None:
    """Write a DataFrame to a Parquet file."""
    df._data.write_parquet(path)


def write_csv(df: DataFrame[Any], path: str, **kwargs: Any) -> None:
    """Write a DataFrame to a CSV file."""
    df._data.write_csv(path, **kwargs)
