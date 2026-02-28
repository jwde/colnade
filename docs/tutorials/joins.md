# Joins

This tutorial demonstrates joining DataFrames from different schemas and flattening the result with `cast_schema`.

!!! tip "Runnable example"
    The complete code is in [`examples/joins.py`](https://github.com/jwde/colnade/blob/main/examples/joins.py).

## Define schemas

```python
import colnade as cn

class Users(cn.Schema):
    id: cn.Column[cn.UInt64]
    name: cn.Column[cn.Utf8]
    age: cn.Column[cn.UInt64]

class Orders(cn.Schema):
    id: cn.Column[cn.UInt64]
    user_id: cn.Column[cn.UInt64]
    amount: cn.Column[cn.Float64]

class UserOrders(cn.Schema):
    user_name: cn.Column[cn.Utf8] = cn.mapped_from(Users.name)
    user_id: cn.Column[cn.UInt64] = cn.mapped_from(Users.id)
    amount: cn.Column[cn.Float64]
```

The `UserOrders` output schema uses `mapped_from` to disambiguate — both `Users` and `Orders` have an `id` column.

## Perform the join

```python
joined = users.join(orders, on=Users.id == Orders.user_id)
```

`Users.id == Orders.user_id` creates a `JoinCondition` (not a `BinOp`) because the columns belong to different schemas. The result is `JoinedDataFrame[Users, Orders]`.

## Work with joined data

You can use columns from either schema on a `JoinedDataFrame`:

```python
# Filter using columns from either schema
big_orders = joined.filter(Orders.amount >= 150)
young_users = joined.filter(Users.age < 30)
```

## Flatten to output schema

```python
result = joined.cast_schema(UserOrders)
```

`cast_schema` resolves each target column:

- `user_name` → `mapped_from(Users.name)` → selects `name` from the left (Users) side
- `user_id` → `mapped_from(Users.id)` → selects `id` from the left side
- `amount` → name matches `Orders.amount` (unambiguous)

The result is `DataFrame[UserOrders]` with columns `["user_name", "user_id", "amount"]`.

## Filter then cast

You can chain operations on the joined data before casting:

```python
result = (
    joined
    .filter(Orders.amount >= 150)
    .cast_schema(UserOrders)
)
```
