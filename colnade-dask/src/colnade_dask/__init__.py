"""Colnade Dask backend adapter."""

from colnade_dask.adapter import DaskBackend
from colnade_dask.io import (
    from_dict,
    from_rows,
    read_csv,
    read_parquet,
    scan_csv,
    scan_parquet,
    write_csv,
    write_parquet,
)

__all__ = [
    "DaskBackend",
    "from_dict",
    "from_rows",
    "read_csv",
    "read_parquet",
    "scan_csv",
    "scan_parquet",
    "write_csv",
    "write_parquet",
]
