# Generic Functions

This tutorial demonstrates how to write schema-polymorphic utility functions that work with any schema while preserving the type parameter.

!!! tip "Runnable example"
    The complete code is in [`examples/generic_functions.py`](https://github.com/jwde/colnade/blob/main/examples/generic_functions.py).

## The TypeVar pattern

Import the schema-bound TypeVar `S`:

```python
from colnade import DataFrame
from colnade.schema import S
```

`S` is `TypeVar("S", bound=Schema)`. Functions parameterized by `S` accept any schema and preserve it in the return type.

## Writing generic functions

```python
def first_n(df: DataFrame[S], n: int) -> DataFrame[S]:
    """Return the first n rows. Works with any schema."""
    return df.head(n)

def drop_null_rows(df: DataFrame[S]) -> DataFrame[S]:
    """Drop all rows with any null values."""
    return df.drop_nulls()

def count_rows(df: DataFrame[S]) -> int:
    """Count rows in any typed DataFrame."""
    return len(df)
```

## Type preservation in action

The type checker knows the concrete schema at each call site:

```python
users: DataFrame[Users] = read_parquet("users.parquet", Users)
products: DataFrame[Products] = read_parquet("products.parquet", Products)

# first_n preserves the input schema
top_users = first_n(users, 10)          # DataFrame[Users]
top_products = first_n(products, 5)     # DataFrame[Products]

# Works across different schemas
clean_users = drop_null_rows(users)     # DataFrame[Users]
clean_products = drop_null_rows(products)  # DataFrame[Products]
```

## Pipeline utilities

Generic functions compose naturally in pipelines:

```python
def standard_cleanup(df: DataFrame[S]) -> DataFrame[S]:
    """Standard cleanup: drop nulls, deduplicate, sort by first column."""
    return df.drop_nulls()

result = standard_cleanup(users)  # DataFrame[Users]
```

The key benefit: you write the function once, and the type checker verifies it works correctly for every schema it's called with.
