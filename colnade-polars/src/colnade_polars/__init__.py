"""Colnade Polars backend adapter."""

from importlib.metadata import version as _version

__version__: str = _version("colnade-polars")

from colnade_polars.adapter import PolarsBackend
from colnade_polars.conversion import map_colnade_dtype, map_polars_dtype
from colnade_polars.io import (
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
    "PolarsBackend",
    "map_colnade_dtype",
    "map_polars_dtype",
    "from_dict",
    "from_rows",
    "read_csv",
    "read_parquet",
    "scan_csv",
    "scan_parquet",
    "write_csv",
    "write_parquet",
]
