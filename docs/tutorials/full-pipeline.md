# Full Pipeline

This tutorial demonstrates a complete ETL pipeline: read data, clean nulls, filter, join, aggregate, cast to output schema, and write results.

!!! tip "Runnable example"
    The complete code is in [`examples/full_pipeline.py`](https://github.com/jwde/colnade/blob/main/examples/full_pipeline.py).

## Define schemas

```python
from colnade import Column, Schema, UInt64, Float64, Utf8, mapped_from

class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]

class Orders(Schema):
    id: Column[UInt64]
    user_id: Column[UInt64]
    amount: Column[Float64]

class UserRevenue(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    user_id: Column[UInt64] = mapped_from(Users.id)
    total_amount: Column[Float64]
```

## Step 1: Read typed data

```python
from colnade_polars.io import read_parquet

users = read_parquet("users.parquet", Users)
orders = read_parquet("orders.parquet", Orders)
```

Both `read_parquet` calls validate the data against their schema and attach the Polars backend.

## Step 2: Clean nulls

```python
users_clean = users.with_columns(
    Users.score.fill_null(0.0).alias(Users.score)
)
```

## Step 3: Filter

```python
active_users = users_clean.filter(Users.age >= 25)
```

## Step 4: Join

```python
joined = active_users.join(orders, on=Users.id == Orders.user_id)
```

## Step 5: Cast to output schema

```python
user_orders = joined.cast_schema(UserRevenue, mapping={
    UserRevenue.total_amount: Orders.amount,
})
```

`mapped_from` resolves `user_name` and `user_id`. The explicit mapping handles `total_amount` → `Orders.amount`.

## Step 6: Sort and write

```python
from colnade_polars import write_parquet

result = user_orders.sort(UserRevenue.total_amount, descending=True)
write_parquet(result, "user_revenue.parquet")
```

## Step 7: Verify

```python
restored = read_parquet("user_revenue.parquet", UserRevenue)
restored.validate()  # confirms data matches schema
```

## Pipeline summary

```
read_parquet(Users) → fill_null → filter(age >= 25)
                                        ↓
read_parquet(Orders) ──────────→ join(Users.id == Orders.user_id)
                                        ↓
                              cast_schema(UserRevenue)
                                        ↓
                              sort(total_amount desc)
                                        ↓
                              write_parquet(result)
```

Every step in this pipeline is type-checked. The type checker verifies that column references exist, schema types match at boundaries, and `mapped_from` assignments are type-compatible.
