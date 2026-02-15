# Installation

## Requirements

- Python 3.10 or higher
- A supported type checker: [ty](https://github.com/astral-sh/ty), mypy, or pyright

## Install with pip

```bash
pip install colnade colnade-polars
```

`colnade` provides the core abstraction layer. Install the backend adapter for your engine:

| Backend | Install |
|---------|---------|
| Polars | `pip install colnade-polars` |
| Pandas | `pip install colnade-pandas` |
| Dask | `pip install colnade-dask` |

## Install with uv

```bash
uv add colnade colnade-polars
```

Or substitute `colnade-pandas` / `colnade-dask` for your preferred backend.

## Install from source

```bash
git clone https://github.com/jwde/colnade.git
cd colnade
uv sync
```

## Verify installation

```python
from colnade import Column, Schema, UInt64, Utf8
from colnade_polars import PolarsBackend

class TestSchema(Schema):
    id: Column[UInt64]
    name: Column[Utf8]

print("Colnade installed successfully!")
print(f"Columns: {list(TestSchema._columns.keys())}")
```

## Type checker setup

Colnade works with any Python type checker. Here's how to set up the most common ones.

### ty (recommended)

```bash
pip install ty
ty check src/
```

### mypy

Add to your `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.10"
strict = true
```

### pyright

Add to your `pyproject.toml`:

```toml
[tool.pyright]
pythonVersion = "3.10"
typeCheckingMode = "strict"
```
