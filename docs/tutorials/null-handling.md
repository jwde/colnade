# Null Handling

This tutorial demonstrates how to work with nullable data in Colnade.

!!! tip "Runnable example"
    The complete code is in [`examples/null_handling.py`](https://github.com/jwde/colnade/blob/main/examples/null_handling.py).

## Setup

```python
from colnade import Column, Float64, Schema, UInt64, Utf8
from colnade_polars import from_dict

class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]

df = from_dict(Users, {
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age": [30, None, 35, None, 40],
    "score": [85.0, 92.5, None, 95.0, None],
})
```

## fill_null — replace nulls with a value

```python
filled = df.with_columns(
    Users.age.fill_null(0).alias(Users.age),
    Users.score.fill_null(0.0).alias(Users.score),
)
```

`fill_null(value)` creates a `FunctionCall` expression. The backend translates it to `pl.col("age").fill_null(0)`. Multiple columns can be filled in a single `with_columns` call.

## drop_nulls — remove rows with nulls

```python
# Drop rows where score is null
clean = df.drop_nulls(Users.score)

# Drop rows where any of the specified columns are null
clean = df.drop_nulls(Users.age, Users.score)
```

## is_null / is_not_null — filter by null status

```python
# Keep only rows with null scores
null_scores = df.filter(Users.score.is_null())

# Keep only rows with non-null scores
valid_scores = df.filter(Users.score.is_not_null())
```

## Combining null handling with other operations

A common pattern: fill nulls, then filter:

```python
result = df.with_columns(
    Users.score.fill_null(0.0).alias(Users.score),
).filter(Users.score > 50)
```

Or combine null checks with value filters:

```python
result = df.filter(
    Users.score.is_not_null() & (Users.score > 50)
)
```

## Full cleanup pipeline

```python
result = (
    df.with_columns(
        Users.score.fill_null(0.0).alias(Users.score),
        Users.age.fill_null(0).alias(Users.age),
    )
    .filter(Users.age > 0)
    .sort(Users.score.desc())
)
```
