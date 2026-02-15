# colnade-pandas

Pandas backend adapter for [Colnade](https://github.com/jwde/colnade).

## Installation

```bash
pip install colnade-pandas
```

## Usage

```python
from colnade import Column, Schema, UInt64, Float64, Utf8
from colnade_pandas import read_parquet

class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]

df = read_parquet("users.parquet", Users)
result = df.filter(Users.age > 25).sort(Users.score.desc())
```

## I/O Functions

- `read_parquet` / `write_parquet`
- `read_csv` / `write_csv`

## Documentation

Full documentation at [colnade.com](https://colnade.com/).
