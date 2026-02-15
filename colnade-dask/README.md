# colnade-dask

Dask backend adapter for [Colnade](https://github.com/jwde/colnade). Supports lazy evaluation and distributed computation.

## Installation

```bash
pip install colnade-dask
```

## Usage

```python
from colnade import Column, Schema, UInt64, Float64, Utf8
from colnade_dask import scan_parquet

class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]

lf = scan_parquet("users.parquet", Users)
result = lf.filter(Users.age > 25).sort(Users.score.desc()).collect()
```

## I/O Functions

- `read_parquet` / `write_parquet` (eager)
- `scan_parquet` / `scan_csv` (lazy)
- `read_csv` / `write_csv`

## Documentation

Full documentation at [colnade.com](https://colnade.com/).
