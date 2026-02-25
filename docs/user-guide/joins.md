# Joins

Colnade supports typed joins between DataFrames with different schemas.

## Join condition

When you compare columns from different schemas with `==`, Colnade creates a `JoinCondition` instead of a regular `BinOp`:

```python
# Same schema → BinOp[Bool] (filter expression)
Users.age == 30

# Different schemas → JoinCondition
Users.id == Orders.user_id
```

## Performing a join

```python
joined = users.join(orders, on=Users.id == Orders.user_id)
# joined is JoinedDataFrame[Users, Orders]
```

The `how` parameter controls the join type:

```python
users.join(orders, on=Users.id == Orders.user_id, how="inner")   # default
users.join(orders, on=Users.id == Orders.user_id, how="left")
users.join(orders, on=Users.id == Orders.user_id, how="outer")
```

## JoinedDataFrame — a transitional type

`JoinedDataFrame[S, S2]` is a **transitional** type. It holds data from two schemas but does not itself satisfy a single schema contract. Before you can use schema-dependent operations like `group_by()`, `head()`, `tail()`, or `sample()`, you must call `cast_schema()` to flatten the join result into a `DataFrame[S3]`.

Available operations on `JoinedDataFrame`:

| Operation | Returns | Description |
|-----------|---------|-------------|
| `filter()` | `JoinedDataFrame[S, S2]` | Filter rows using columns from either schema |
| `sort()` | `JoinedDataFrame[S, S2]` | Sort rows |
| `limit()` | `JoinedDataFrame[S, S2]` | Limit to first n rows |
| `unique()` | `JoinedDataFrame[S, S2]` | Deduplicate by columns |
| `drop_nulls()` | `JoinedDataFrame[S, S2]` | Drop rows with nulls |
| `with_columns()` | `JoinedDataFrame[S, S2]` | Add or overwrite columns |
| `select()` | `DataFrame[Any]` | Select columns (untyped) |
| `cast_schema()` | `DataFrame[S3]` | Flatten to a single schema |
| `lazy()` | `JoinedLazyFrame[S, S2]` | Convert to lazy |

Operations **not** available on joined frames (use `cast_schema()` first):

- `group_by()` — requires a single schema
- `head()`, `tail()`, `sample()` — materialized-only ops on `DataFrame`

```python
# Typical workflow: join → filter → cast_schema → group_by
result = (
    users.join(orders, on=Users.id == Orders.user_id)
    .filter(Orders.amount >= 100)
    .cast_schema(UserOrders)
    .group_by(UserOrders.user_name)
    .agg(UserOrders.amount.sum().alias(UserOrders.amount))
    .cast_schema(UserTotals)
)
```

## Flattening with cast_schema

Use `cast_schema` to flatten a `JoinedDataFrame` into a single-schema `DataFrame`:

```python
class UserOrders(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    user_id: Column[UInt64] = mapped_from(Users.id)
    amount: Column[Float64]

result = joined.cast_schema(UserOrders)
# result is DataFrame[UserOrders]
```

## Disambiguating shared column names

When both schemas have a column with the same name (e.g., both `Users` and `Orders` have `id`), use `mapped_from` to specify which source you want:

```python
class JoinOutput(Schema):
    user_id: Column[UInt64] = mapped_from(Users.id)       # from Users
    order_id: Column[UInt64] = mapped_from(Orders.id)     # from Orders
    amount: Column[Float64]                                # unambiguous name match
```

Without `mapped_from`, ambiguous column names (present in both schemas) are skipped during name matching and will produce a `SchemaError` for missing columns.

## Explicit mapping

For full control, pass an explicit mapping dict:

```python
result = joined.cast_schema(Output, mapping={
    Output.person_name: Users.name,
    Output.total: Orders.amount,
})
```

Explicit mapping takes the highest precedence, overriding both `mapped_from` and name matching.

## Lazy joins

`LazyFrame` joins work the same way but produce `JoinedLazyFrame[S, S2]`:

```python
joined_lazy = users_lazy.join(orders_lazy, on=Users.id == Orders.user_id)
# JoinedLazyFrame[Users, Orders]

result = joined_lazy.cast_schema(UserOrders).collect()
# DataFrame[UserOrders]
```

`JoinedLazyFrame` has the same restrictions as `JoinedDataFrame` — no `group_by()`, `head()`, `tail()`, or `sample()`. Use `cast_schema()` first.
