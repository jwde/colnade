# colnade-polars

Polars backend adapter for [Colnade](https://pypi.org/project/colnade/) — a statically type-safe DataFrame abstraction layer for Python.

## Installation

```bash
pip install colnade colnade-polars
```

## Usage

```python
from colnade import Column, Schema, UInt64, Utf8
from colnade_polars import read_parquet

class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]

df = read_parquet("users.parquet", Users)
# df is DataFrame[Users] — fully type-checked
```

See the [full documentation](https://colnade.com/) for details.
