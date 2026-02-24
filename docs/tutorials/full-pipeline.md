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

class UserOrders(Schema):
    """Intermediate schema after joining users with orders."""
    user_name: Column[Utf8] = mapped_from(Users.name)
    user_id: Column[UInt64] = mapped_from(Users.id)
    amount: Column[Float64] = mapped_from(Orders.amount)

class UserRevenue(Schema):
    """Output schema: per-user revenue summary."""
    user_name: Column[Utf8]
    user_id: Column[UInt64]
    total_amount: Column[Float64]
```

## Step 1: Read typed data

```python
from colnade_polars import read_parquet

users = read_parquet("users.parquet", Users)
orders = read_parquet("orders.parquet", Orders)
```

Both `read_parquet` calls return typed DataFrames with the Polars backend attached. When [validation is enabled](../user-guide/dataframes.md#validation), they also check the data against the schemas.

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

## Step 5: Cast to intermediate schema

```python
user_orders = joined.cast_schema(UserOrders)
```

`mapped_from` resolves `user_name`, `user_id`, and `amount` automatically from the join result.

## Step 6: Aggregate

```python
revenue = (
    user_orders.group_by(UserOrders.user_name, UserOrders.user_id)
    .agg(UserOrders.amount.sum().alias(UserRevenue.total_amount))
    .cast_schema(UserRevenue)
)
```

`group_by().agg()` computes per-user totals. The result is cast to `UserRevenue`.

## Step 7: Sort and write

```python
from colnade_polars import write_parquet

result = revenue.sort(UserRevenue.total_amount, descending=True)
write_parquet(result, "user_revenue.parquet")
```

## Step 8: Verify

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
                              cast_schema(UserOrders)
                                        ↓
                              group_by().agg(sum) → cast_schema(UserRevenue)
                                        ↓
                              sort(total_amount desc)
                                        ↓
                              write_parquet(result)
```

Every step in this pipeline is type-checked. The type checker verifies that column references exist, schema types match at boundaries, and `mapped_from` assignments are type-compatible.
