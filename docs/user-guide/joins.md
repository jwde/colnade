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
users.join(orders, on=Users.id == Orders.user_id, how="cross")
```

## JoinedDataFrame

`JoinedDataFrame[S, S2]` accepts columns from either schema. All schema-preserving operations return `JoinedDataFrame[S, S2]`:

```python
joined.filter(Users.age > 25)           # filter by left schema column
joined.filter(Orders.amount >= 100)     # filter by right schema column
joined.sort(Users.name)                 # sort
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
