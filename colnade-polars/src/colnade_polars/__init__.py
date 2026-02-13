"""Colnade Polars backend adapter."""

from colnade_polars.adapter import PolarsBackend
from colnade_polars.conversion import map_colnade_dtype, map_polars_dtype
from colnade_polars.io import (
    read_csv,
    read_parquet,
    scan_csv,
    scan_parquet,
    write_csv,
    write_parquet,
)

__all__ = [
    "PolarsBackend",
    "map_colnade_dtype",
    "map_polars_dtype",
    "read_csv",
    "read_parquet",
    "scan_csv",
    "scan_parquet",
    "write_csv",
    "write_parquet",
]
