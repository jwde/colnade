"""Read/write operations for Dask backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import dask.dataframe as dd

from colnade import LazyFrame, Row, Schema
from colnade.dataframe import rows_to_dict
from colnade.validation import ValidationLevel, get_validation_level, is_validation_enabled
from colnade_dask.adapter import DaskBackend
from colnade_pandas.conversion import map_colnade_dtype

S = TypeVar("S", bound=Schema)


def _build_pandas_schema(schema: type[S]) -> dict[str, Any]:
    """Build a Pandas dtype dict from a Colnade schema."""
    return {name: map_colnade_dtype(col.dtype) for name, col in schema._columns.items()}


def scan_parquet(path: str, schema: type[S], **kwargs: Any) -> LazyFrame[S]:
    """Lazily scan a Parquet file into a typed LazyFrame backed by Dask."""
    backend = DaskBackend()
    data = dd.read_parquet(path, **kwargs)
    if is_validation_enabled():
        backend.validate_schema(data, schema)
        if get_validation_level() is ValidationLevel.FULL:
            backend.validate_field_constraints(data, schema)
    return LazyFrame(_data=data, _schema=schema, _backend=backend)


def scan_csv(path: str, schema: type[S], **kwargs: Any) -> LazyFrame[S]:
    """Lazily scan a CSV file into a typed LazyFrame backed by Dask.

    Applies the schema's dtype mapping to ensure correct column types.
    """
    backend = DaskBackend()
    pd_schema = _build_pandas_schema(schema)
    data = dd.read_csv(path, dtype=pd_schema, **kwargs)
    if is_validation_enabled():
        backend.validate_schema(data, schema)
        if get_validation_level() is ValidationLevel.FULL:
            backend.validate_field_constraints(data, schema)
    return LazyFrame(_data=data, _schema=schema, _backend=backend)


def from_dict(
    schema: type[S],
    data: dict[str, Sequence[Any]],
) -> LazyFrame[S]:
    """Create a typed LazyFrame from a columnar dict.

    Returns a ``LazyFrame`` because Dask is inherently lazy — use
    ``.collect()`` to materialize.  The schema drives dtype coercion so
    plain Python values (``[1, 2, 3]``) are cast to the correct native
    types (e.g. ``UInt64``).
    """
    backend = DaskBackend()
    native = backend.from_dict(data, schema)
    if is_validation_enabled():
        backend.validate_schema(native, schema)
        if get_validation_level() is ValidationLevel.FULL:
            backend.validate_field_constraints(native, schema)
    return LazyFrame(_data=native, _schema=schema, _backend=backend)


def from_rows(
    schema: type[S],
    rows: Sequence[Row[S]],
) -> LazyFrame[S]:
    """Create a typed LazyFrame from ``Row[S]`` instances.

    Returns a ``LazyFrame`` because Dask is inherently lazy — use
    ``.collect()`` to materialize.  The type checker verifies that rows
    match the schema — passing ``Orders.Row`` where ``Users.Row`` is
    expected is a static error.
    """
    data = rows_to_dict(rows, schema)
    return from_dict(schema, data)


def write_parquet(df: Any, path: str, **kwargs: Any) -> None:
    """Write a DataFrame or LazyFrame to a Parquet file."""
    df._data.to_parquet(path, **kwargs)


def write_csv(df: Any, path: str, **kwargs: Any) -> None:
    """Write a DataFrame or LazyFrame to a CSV file."""
    df._data.to_csv(path, index=False, single_file=True, **kwargs)
