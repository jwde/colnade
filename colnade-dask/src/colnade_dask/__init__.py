"""Colnade Dask backend adapter."""

from importlib.metadata import version as _version

__version__: str = _version("colnade-dask")

from colnade_dask.adapter import DaskBackend
from colnade_dask.io import (
    scan_csv,
    scan_parquet,
    write_csv,
    write_parquet,
)

__all__ = [
    "DaskBackend",
    "scan_csv",
    "scan_parquet",
    "write_csv",
    "write_parquet",
]
