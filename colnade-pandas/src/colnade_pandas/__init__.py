"""Colnade Pandas backend adapter."""

from importlib.metadata import version as _version

__version__: str = _version("colnade-pandas")

from colnade_pandas.adapter import PandasBackend
from colnade_pandas.conversion import map_colnade_dtype, map_pandas_dtype
from colnade_pandas.io import (
    from_dict,
    from_rows,
    read_csv,
    read_parquet,
    write_csv,
    write_parquet,
)

__all__ = [
    "PandasBackend",
    "map_colnade_dtype",
    "map_pandas_dtype",
    "from_dict",
    "from_rows",
    "read_csv",
    "read_parquet",
    "write_csv",
    "write_parquet",
]
